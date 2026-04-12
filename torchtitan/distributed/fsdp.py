# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed._composable.replicate_with_fsdp import replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Shard

from torchtitan.tools.logging import logger


def get_fsdp_reshard_after_forward_policy(
    reshard_after_forward_policy: str, pp_enabled: bool
) -> bool:
    """Resolve fsdp_reshard_after_forward policy string to a boolean.

    Args:
        reshard_after_forward_policy: One of "always", "never", or "default".
        pp_enabled: Whether pipeline parallelism is enabled.

    Returns:
        Boolean indicating whether to reshard after forward.
    """
    match reshard_after_forward_policy:
        case "always":
            return True
        case "never":
            return False
        case "default":
            # For PP, by default do not reshard after forward to avoid per-microbatch
            # all-gathers, which can be expensive and non-overlapped
            return not pp_enabled
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )


def disable_fsdp_gradient_division(model: nn.Module) -> None:
    """
    Disable FSDP's automatic gradient division for all FSDP modules.

    Set gradient_divide_factor=1.0 to disable FSDP's automatic gradient division.
    We handle gradient scaling ourselves in the training loop with global token count.

    Note: This also works for ReplicateModule since it inherits from FSDPModule.

    Args:
        model: The model containing FSDP-wrapped or Replicate-wrapped modules
    """
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.0)


def apply_replicate_to_model(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
) -> None:
    """
    Apply data parallelism via replication (HSDP replicate mode) to the model.

    This function applies replicate to:
    - tok_embeddings (if present)
    - Each transformer block in model.layers
    - norm and output layers (if both present)
    - The root model

    Args:
        model: The model to apply replication to. Expected to have attributes:
            - layers: ModuleDict or Sequential of transformer blocks
            - tok_embeddings: Optional[nn.Module]
            - norm: Optional[nn.Module]
            - output: Optional[nn.Module]
        dp_mesh: The device mesh for data parallelism.
        param_dtype: The data type for model parameters.
        reduce_dtype: The data type for reduction operations.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    replicate_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}

    tok_embeddings = getattr(model, "tok_embeddings", None)
    if tok_embeddings is not None:
        replicate(tok_embeddings, **replicate_config)

    layers = getattr(model, "layers", None)
    if layers is not None:
        for transformer_block in layers.values():
            replicate(transformer_block, **replicate_config)

    norm = getattr(model, "norm", None)
    output = getattr(model, "output", None)
    if norm is not None and output is not None:
        replicate([norm, output], **replicate_config)

    replicate(model, **replicate_config)

    # Disable Replicate's automatic gradient division (ReplicateModule inherits from FSDPModule)
    disable_fsdp_gradient_division(model)

    logger.info("Applied replicate to the model")


def apply_fsdp_to_model(
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
    prefetch: bool = True,
) -> None:
    """
    Apply data parallelism (via FSDP2) to the model.

    This function supports both standard Transformer models and MoE (Mixture of Experts)
    models. For MoE models with expert parallelism (EP > 1), it uses different FSDP
    meshes for expert and non-expert parameters.

    Args:
        model: The model to apply FSDP to. Expected to have attributes:
            - layers: ModuleDict or Sequential of transformer blocks
            - tok_embeddings: Optional[nn.Module]
            - norm: Optional[nn.Module]
            - output: Optional[nn.Module]
            For MoE support, each layer may have:
            - moe_enabled: bool
            - moe: Module with experts attribute
        dp_mesh: The device mesh for FSDP data parallelism.
        param_dtype: The data type for model parameters.
        reduce_dtype: The data type for reduction operations.
        pp_enabled: Whether pipeline parallelism is enabled.
        cpu_offload: Whether to offload model parameters to CPU. Defaults to False.
        reshard_after_forward_policy: The policy for resharding after forward.
            Options: "default", "never", "always". Defaults to "default".
        ep_degree: Expert parallelism degree. Defaults to 1 (no EP).
        edp_mesh: Device mesh for expert data parallelism (used when ep_degree > 1).
        gradient_divide_factor: Optional factor for gradient division.
        prefetch: Whether to set up explicit prefetching for EP. Defaults to True.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )

    enable_weight_tying = getattr(model, "enable_weight_tying", False)
    tok_embeddings = getattr(model, "tok_embeddings", None)
    norm = getattr(model, "norm", None)
    output = getattr(model, "output", None)

    if enable_weight_tying:
        # When weights are tied, tok_embeddings and output share the same parameter.
        # Group them together in one FSDP unit to avoid duplicate all-gathers.
        modules = [m for m in (tok_embeddings, norm, output) if m is not None]
        fully_shard(
            modules,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )
    else:
        if tok_embeddings is not None:
            fully_shard(
                tok_embeddings,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
        # As an optimization, do not reshard_after_forward the last layers by default
        # since FSDP would prefetch them immediately after the forward pass
        if norm is not None and output is not None:
            fully_shard(
                [norm, output],
                **fsdp_config,
                reshard_after_forward=reshard_after_forward_policy == "always",
            )

    layers = getattr(model, "layers", None)
    if layers is not None:
        for transformer_block in layers.values():
            # Check if this is an MoE layer
            moe_enabled = getattr(transformer_block, "moe_enabled", False)

            if moe_enabled:
                _apply_fsdp_to_moe_block(
                    transformer_block,
                    fsdp_config,
                    dp_mesh,
                    edp_mesh,
                    ep_degree,
                    reshard_after_forward,
                )
            else:
                fully_shard(
                    transformer_block,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                )

    fully_shard(model, **fsdp_config)

    # Disable FSDP's automatic gradient division for all FSDP modules
    disable_fsdp_gradient_division(model)

    # Set up explicit prefetching when EP is enabled
    if prefetch and ep_degree > 1 and layers is not None:
        _setup_fsdp_prefetching(model, layers)


def _apply_fsdp_to_moe_block(
    transformer_block: nn.Module,
    fsdp_config: dict[str, Any],
    dp_mesh: DeviceMesh,
    edp_mesh: DeviceMesh | None,
    ep_degree: int,
    reshard_after_forward: bool,
) -> None:
    """
    Apply FSDP to an MoE transformer block with expert-aware sharding.

    For MoE layers, we use shard_placement_fn to apply different FSDP mesh and
    shard placement to different parameters:
    - When EP > 1: routed experts use edp_mesh, other params use dp_mesh
    - When EP = 1: all params use the same FSDP mesh, but experts may use Shard(1)
      when FSDP degree > num_experts to avoid padding
    """
    moe = getattr(transformer_block, "moe", None)
    if moe is None or not hasattr(moe, "experts"):
        # Fallback: treat as regular block if moe structure is unexpected
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
        return

    expert_params = set(moe.experts.parameters())
    num_experts = moe.experts.num_experts

    if ep_degree > 1:
        assert edp_mesh is not None
        efsdp_ep_size = edp_mesh["efsdp"].size() * ep_degree
    else:
        efsdp_ep_size = fsdp_config["mesh"].size()

    if efsdp_ep_size > num_experts:
        expert_shard_placement = Shard(1)
    else:
        expert_shard_placement = Shard(0)

    # When ep_degree == 1 and no Shard(1) override needed, skip
    # shard_placement_fn entirely for simplicity
    if ep_degree == 1 and expert_shard_placement == Shard(0):
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    elif ep_degree == 1:
        # ep_degree == 1 but need Shard(1) for experts to avoid padding
        def _experts_shard_placement_fn(
            param: nn.Parameter,
            _expert_params: set = expert_params,
        ) -> Shard | None:
            if param in _expert_params:
                return Shard(1)
            return None

        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
            shard_placement_fn=_experts_shard_placement_fn,
        )
    else:
        # ep_degree > 1: per-param mesh
        # Note: Per-parameter mesh selection requires ShardPlacementResult which is
        # available in newer PyTorch versions. For now, we use a workaround by
        # applying FSDP separately to expert and non-expert modules.
        # TODO: Add proper per-param mesh support when PyTorch API is available.

        # Separate expert and non-expert parameters
        expert_modules = list(moe.experts.modules())
        non_expert_modules = [
            m
            for m in transformer_block.modules()
            if m not in expert_modules and m != transformer_block
        ]

        # Apply FSDP to experts with edp_mesh
        expert_fsdp_config = dict(fsdp_config)
        expert_fsdp_config["mesh"] = edp_mesh

        for expert_module in moe.experts:
            fully_shard(
                expert_module,
                **expert_fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

        # Apply FSDP to the rest of the transformer block
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )


def _setup_fsdp_prefetching(
    model: nn.Module,
    layers: nn.ModuleDict | nn.Sequential,
) -> None:
    """
    Set up explicit prefetching for FSDP modules when EP is enabled.

    D2H syncs in EP could interfere with implicit prefetching in FSDP,
    so we set up explicit prefetching.
    """
    tok_embeddings = getattr(model, "tok_embeddings", None)
    norm = getattr(model, "norm", None)
    output = getattr(model, "output", None)

    transformer_blocks = list(layers.values())
    if len(transformer_blocks) == 0:
        return

    next_transformer_blocks = transformer_blocks[1:] + [None]

    # Forward prefetching
    if tok_embeddings is not None:
        tok_embeddings.set_modules_to_forward_prefetch([transformer_blocks[0]])

    for transformer_block, next_transformer_block in zip(
        transformer_blocks, next_transformer_blocks
    ):
        if next_transformer_block is not None:
            transformer_block.set_modules_to_forward_prefetch([next_transformer_block])
        elif norm is not None and output is not None:
            transformer_block.set_modules_to_forward_prefetch([norm, output])

    # Backward prefetching
    reversed_transformer_blocks = list(reversed(transformer_blocks))
    prev_transformer_blocks = reversed_transformer_blocks[1:] + [None]

    if output is not None and norm is not None:
        output.set_modules_to_backward_prefetch([reversed_transformer_blocks[0]])

    for transformer_block, prev_transformer_block in zip(
        reversed_transformer_blocks, prev_transformer_blocks
    ):
        if prev_transformer_block is not None:
            transformer_block.set_modules_to_backward_prefetch([prev_transformer_block])
        elif tok_embeddings is not None:
            transformer_block.set_modules_to_backward_prefetch([tok_embeddings])
