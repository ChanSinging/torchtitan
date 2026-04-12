## Why

当前 torchtitan 中的 FSDP2 应用逻辑分散在各个模型的 `parallelize.py` 文件中（llama3、llama4、deepseek_v3 等），存在大量重复代码。每个模型都需要独立实现 `apply_fsdp`、`apply_replicate` 等函数，导致维护困难和一致性风险。此外，FSDP2 的 MoE 支持（expert parallelism、shard_placement_fn 等）仅在 llama4 中完整实现，其他模型无法复用。通过将 FSDP2 应用逻辑提取到通用组件中，可以提高代码复用性、降低新模型接入门槛，并确保并行策略的一致性。

## What Changes

- **提取通用 FSDP2 应用函数**：将 `apply_fsdp`、`apply_replicate`、`disable_fsdp_gradient_division` 从模型特定的 `parallelize.py` 移动到 `torchtitan/distributed/fsdp.py`
- **统一 MoE FSDP2 支持**：将 llama4 中的 MoE FSDP2 逻辑（shard_placement_fn、专家并行支持）提取为通用函数，供所有 MoE 模型使用
- **标准化模型接口**：定义 FSDP2 应用所需的模型结构约定（如 `model.layers`、`model.tok_embeddings`、`model.moe_enabled` 等）
- **重构现有模型**：更新 llama3、llama4、deepseek_v3、gpt_oss、flux 等模型的 `parallelize.py`，使其使用通用 FSDP2 函数
- **添加单元测试**：为通用 FSDP2 函数添加 CPU 级别的单元测试

## Capabilities

### New Capabilities
- `fsdp2-generalization`: 通用 FSDP2 应用框架，支持标准 Transformer 和 MoE 模型

### Modified Capabilities
- （无现有 spec 需要修改，此为纯重构变更）

## Impact

- **主要修改文件**：
  - `torchtitan/distributed/fsdp.py`：添加通用 FSDP2 应用函数
  - `torchtitan/models/*/parallelize.py`：重构以使用通用函数
- **API 变更**：新增通用函数 `apply_fsdp_to_model()`、`apply_replicate_to_model()` 等
- **无破坏性变更**：现有模型配置和训练脚本保持兼容
- **测试要求**：需要验证所有模型（llama3、llama4、deepseek_v3 等）的数值一致性
