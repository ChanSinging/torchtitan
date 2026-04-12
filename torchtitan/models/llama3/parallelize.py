# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D parallelisms (except pipeline parallelism) and various
# training techniques (e.g. activation checkpointing and compile) to the Llama model.

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.components.quantization.float8 import find_float8_linear_config
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.compile import apply_compile_dense
from torchtitan.distributed.context_parallel import apply_cp_to_attention_module
from torchtitan.distributed.fsdp import apply_fsdp_to_model, apply_replicate_to_model
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp, NoParallel
from torchtitan.models.llama3.model import Llama3Model
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger


def parallelize_llama(
    model: Llama3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    # TODO: TP currently cannot handle uneven seq_len because we set
    #       `use_local_output=True` to use plain Tensors for legacy reasons.
    #       Need to revisit this.
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    if parallel_dims.tp_enabled:
        float8_config = find_float8_linear_config(model_converters.converters)
        enable_float8_linear = float8_config is not None
        float8_is_rowwise = float8_config is not None and float8_config.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        # For now, float8 all-gather with TP is only supported for tensorwise
        # float8 scaling recipes. For rowwise recipes, we use regular TP and
        # all-gather happens in high precision.
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise

        enable_sp = parallelism.enable_sequence_parallel

        tp_mesh = parallel_dims.get_mesh("tp")
        apply_tp(
            model,
            tp_mesh,
            enable_loss_parallel=not parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
            enable_cp=parallel_dims.cp_enabled,
            enable_sp=enable_sp,
        )
        maybe_enable_async_tp(parallelism, compile_config, tp_mesh)

    if parallel_dims.cp_enabled:
        apply_cp_to_attention_module(
            # pyrefly: ignore [missing-attribute, not-callable]
            [block.attention.inner_attention for block in model.layers.values()],
            parallel_dims.get_mesh("cp"),
        )

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if model_compile_enabled:
        apply_compile_dense(model, compile_config)

    names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    dp_mesh = parallel_dims.get_mesh(names)
    apply_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
    )

    logger.info("Applied fully_shard to the model")

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the model")

    return model


def apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    enable_loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    enable_cp: bool = False,
    enable_sp: bool = True,
):
    """Apply tensor parallelism."""
    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    sp_layout = Shard(1) if enable_sp else Replicate()
    embed_plan = RowwiseParallel(
        input_layouts=Replicate(),
        output_layouts=sp_layout,
        use_local_output=enable_sp,
    )

    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": embed_plan,
            "norm": SequenceParallel() if enable_sp else NoParallel(),
            "output": ColwiseParallel(
                input_layouts=sp_layout,
                output_layouts=Shard(-1) if enable_loss_parallel else Replicate(),
                use_local_output=not enable_loss_parallel,
            ),
        },
    )

    # Parallel styles used for transformer block linear weights and their
    # inputs may be different for float8 linears with tensorwise scaling.
    if enable_float8_tensorwise_tp:
        # TODO(vkuzo): add the items below to __init__.py of torchao.float8 and import from there
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )

        rowwise_parallel, colwise_parallel, prepare_module_input = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
        )
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    norm_plan = SequenceParallel() if enable_sp else NoParallel()
    rowwise_output_plan = rowwise_parallel(
        output_layouts=sp_layout, use_local_output=enable_sp
    )

    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        # pyrefly: ignore [no-matching-overload]
        layer_plan = {
            "attention_norm": norm_plan,
            "attention": prepare_module_input(
                input_layouts=(sp_layout, None, None, None),
                desired_input_layouts=(Replicate(), None, None, None),
            ),
            "attention.wq": colwise_parallel(),
            "attention.wk": colwise_parallel(),
            "attention.wv": colwise_parallel(),
            "attention.wo": rowwise_output_plan,
            "ffn_norm": norm_plan,
            "feed_forward": prepare_module_input(
                input_layouts=(sp_layout,),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": colwise_parallel(),
            "feed_forward.w2": rowwise_output_plan,
            "feed_forward.w3": colwise_parallel(),
        }

        parallelize_module(
            # pyrefly: ignore [bad-argument-type]
            module=transformer_block,
            device_mesh=tp_mesh,
            # pyrefly: ignore [bad-argument-type]
            parallelize_plan=layer_plan,
        )

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}"
        "Tensor Parallelism to the model"
    )


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
    """
    Apply data parallelism (via FSDP2) to the model.

    This is a model-specific wrapper around `apply_fsdp_to_model` from
    `torchtitan.distributed.fsdp`.

    Args:
        model: The model to apply data parallelism to.
        dp_mesh: The device mesh to use for data parallelism.
        param_dtype: The data type to use for model parameters.
        reduce_dtype: The data type to use for reduction operations.
        pp_enabled: Whether pipeline parallelism is enabled.
        cpu_offload: Whether to offload model parameters to CPU. Defaults to False.
        reshard_after_forward_policy: The policy for resharding after forward pass.
            Defaults to "default".
    """
    apply_fsdp_to_model(
        model,
        dp_mesh,
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        pp_enabled=pp_enabled,
        cpu_offload=cpu_offload,
        reshard_after_forward_policy=reshard_after_forward_policy,
    )


def apply_replicate(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
):
    """
    Apply data parallelism via replication (HSDP) to the model.

    This is a model-specific wrapper around `apply_replicate_to_model` from
    `torchtitan.distributed.fsdp`.

    Args:
        model: The model to apply replication to.
        dp_mesh: The device mesh for data parallelism.
        param_dtype: The data type for model parameters.
        reduce_dtype: The data type for reduction operations.
    """
    apply_replicate_to_model(
        model,
        dp_mesh,
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
    )
