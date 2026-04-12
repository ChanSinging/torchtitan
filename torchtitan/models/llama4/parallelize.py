# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleInputOutput,
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
from torchtitan.distributed.compile import apply_compile_sparse
from torchtitan.distributed.context_parallel import apply_cp_to_attention_module
from torchtitan.distributed.expert_parallel import (
    DeepEPExpertParallel,
    ExpertParallel,
    ExpertTensorParallel,
    ReordererSequenceParallel,
    TensorParallel,
    TorchAOExpertParallel,
)
from torchtitan.distributed.fsdp import apply_fsdp_to_model
from torchtitan.distributed.tensor_parallel import (
    ColwiseParallelWithGradPlacement,
    maybe_enable_async_tp,
    NoParallel,
)
from torchtitan.models.llama4.model import Llama4Model
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger


def parallelize_llama(
    model: Llama4Model,
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

    tp_mesh = None
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
        apply_non_moe_tp(
            model,
            tp_mesh,
            enable_loss_parallel=not parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
            enable_sp=enable_sp,
        )
        maybe_enable_async_tp(parallelism, compile_config, tp_mesh)

    # Check if using DeepEP/HybridEP for MoE communication
    comm_backend = parallelism.expert_parallel_comm_backend
    if comm_backend in ("deepep", "hybridep"):
        if not parallel_dims.ep_enabled:
            raise ValueError(
                f"{comm_backend.upper()} requires expert parallelism (ep_degree > 1). "
                "Please set expert_parallel_degree > 1 or use standard communication backend."
            )
        if parallel_dims.etp_enabled:
            raise NotImplementedError(
                f"{comm_backend.upper()} with Expert Tensor Parallelism (ETP) is not supported yet. "
                "Please set expert_tensor_parallel_degree=1 or use standard communication backend."
            )

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        from torchtitan.components.quantization import find_pad_multiple

        pad_multiple = find_pad_multiple(model_converters.converters)

        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            etp_mesh=parallel_dims.get_optional_mesh("etp"),
            ep_etp_mesh=parallel_dims.get_optional_mesh(["ep", "etp"]),
            comm_backend=comm_backend,
            hybridep_non_blocking_expert_capacity_factor=parallelism.hybridep_non_blocking_expert_capacity_factor,
            pad_multiple=pad_multiple,
        )

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
        apply_compile_sparse(model, compile_config, parallel_dims.ep_enabled)

    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

    edp_mesh = None
    if parallel_dims.ep_enabled:
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

    apply_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
        gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
    )

    logger.info("Applied fully_shard to the model")

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the model")

    return model


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    enable_loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
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
        }
        # pyrefly: ignore [missing-attribute]
        if not transformer_block.moe_enabled:
            layer_plan.update(
                {
                    "feed_forward": prepare_module_input(
                        input_layouts=(sp_layout,),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "feed_forward.w1": colwise_parallel(),
                    "feed_forward.w2": rowwise_output_plan,
                    "feed_forward.w3": colwise_parallel(),
                }
            )

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
    ep_degree: int = 1,
    edp_mesh: DeviceMesh | None = None,
    gradient_divide_factor: int | None = None,
):
    """
    Apply data parallelism (via FSDP2) to the model.

    This is a model-specific wrapper around `apply_fsdp_to_model` from
    `torchtitan.distributed.fsdp`. It supports both standard Transformer
    and MoE (Mixture of Experts) models.

    Args:
        model: The model to apply data parallelism to.
        dp_mesh: The device mesh to use for data parallelism.
        param_dtype: The data type to use for model parameters.
        reduce_dtype: The data type to use for reduction operations.
        pp_enabled: Whether pipeline parallelism is enabled.
        cpu_offload: Whether to offload model parameters to CPU. Defaults to False.
        reshard_after_forward_policy: The policy for resharding after forward pass.
            Defaults to "default".
        ep_degree: Expert parallelism degree. Defaults to 1 (no EP).
        edp_mesh: Device mesh for expert data parallelism (used when ep_degree > 1).
        gradient_divide_factor: Optional factor for gradient division.
    """
    apply_fsdp_to_model(
        model,
        dp_mesh,
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        pp_enabled=pp_enabled,
        cpu_offload=cpu_offload,
        reshard_after_forward_policy=reshard_after_forward_policy,
        ep_degree=ep_degree,
        edp_mesh=edp_mesh,
        gradient_divide_factor=gradient_divide_factor,
    )


def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    etp_mesh: DeviceMesh | None,
    ep_etp_mesh: DeviceMesh | None,
    comm_backend: str = "standard",
    hybridep_non_blocking_expert_capacity_factor: float | None = None,
    pad_multiple: int | None = None,
):
    assert ep_mesh is not None or tp_mesh is not None

    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        # pyrefly: ignore [missing-attribute]
        if not transformer_block.moe_enabled:
            continue

        if tp_mesh is not None:
            moe_layer_plan = {
                # input / output sharding on the seqlen dim
                # all-gather for input, reduce-scatter for output
                "moe": PrepareModuleInputOutput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                    # Keep input as a DTensor from SequenceParallel, do not wrap with to_local.
                    use_local_input=False,
                    output_layouts=(Partial(),),
                    desired_output_layouts=(Shard(1),),
                ),
                # replicate computation for the router
                "moe.router.gate": NoParallel(
                    local_output_grad_placements=(Partial(),),
                ),
            }
            if ep_mesh is not None and etp_mesh is None:
                # If TP is borrowed for EP, then split the tokens across TP ranks so that
                # the reorderer, the all-to-all comms, and routed experts computation
                # are effectively running Sequence Parallel (split along the folded bs*slen dim)
                # pyrefly: ignore [no-matching-overload]
                moe_layer_plan.update({"moe.reorderer": ReordererSequenceParallel()})
            # pyrefly: ignore [missing-attribute]
            if transformer_block.moe.shared_experts is not None:
                # Use ColwiseParallelWithGradPlacement to keep d_x as Partial in
                # backward (avoids the all-reduce that from_local(Replicate)
                # would otherwise trigger). For w2, output_layouts=Partial()
                # skips the Partial→Replicate all-reduce in forward. The
                # reduction happens once at the MoE output boundary
                # (PrepareModuleInputOutput).
                # pyrefly: ignore [no-matching-overload]
                moe_layer_plan.update(
                    {
                        "moe.shared_experts.w1": ColwiseParallelWithGradPlacement(
                            local_input_grad_placements=(Partial(),)
                        ),
                        "moe.shared_experts.w2": RowwiseParallel(
                            output_layouts=Partial(),
                        ),
                        "moe.shared_experts.w3": ColwiseParallelWithGradPlacement(
                            local_input_grad_placements=(Partial(),)
                        ),
                    }
                )
            parallelize_module(
                # pyrefly: ignore [bad-argument-type]
                module=transformer_block,
                device_mesh=tp_mesh,
                # pyrefly: ignore [bad-argument-type]
                parallelize_plan=moe_layer_plan,
            )

        experts_mesh, experts_plan = None, None
        if ep_mesh is None:
            assert ep_etp_mesh is None
            experts_mesh = tp_mesh
            # input Replicate, output Partial
            experts_plan = TensorParallel()
        elif tp_mesh is None or etp_mesh is None:
            assert ep_etp_mesh is None
            experts_mesh = ep_mesh
            if comm_backend in ("deepep", "hybridep"):
                if comm_backend == "deepep" and pad_multiple is not None:
                    raise ValueError(
                        "DeepEP does not support pad_multiple. "
                        "Use hybridep or standard comm backend instead."
                    )
                # pyrefly: ignore [missing-attribute]
                score_before_experts = transformer_block.moe.score_before_experts

                experts_plan = DeepEPExpertParallel(
                    score_before_experts=score_before_experts,
                    comm_backend=comm_backend,
                    hybridep_non_blocking_expert_capacity_factor=hybridep_non_blocking_expert_capacity_factor,
                    pad_multiple=pad_multiple,
                )
                logger.info(f"Applying {comm_backend.upper()} to MoE layer")
            elif pad_multiple is not None:
                experts_plan = TorchAOExpertParallel(pad_multiple)
            else:
                # input / output sharding on the batch / tokens dim
                experts_plan = ExpertParallel()
        else:
            if pad_multiple is not None:
                raise NotImplementedError(
                    "Quantized grouped GEMMs (FP8/MXFP8) with Expert Tensor "
                    "Parallelism (ETP) is not yet supported. "
                    "Please use EP without ETP."
                )
            experts_mesh = ep_etp_mesh
            experts_plan = ExpertTensorParallel()

        parallelize_module(
            # pyrefly: ignore [missing-attribute]
            module=transformer_block.moe.experts,
            device_mesh=experts_mesh,
            parallelize_plan=experts_plan,
        )
