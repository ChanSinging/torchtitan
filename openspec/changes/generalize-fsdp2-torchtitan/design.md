## Context

当前 torchtitan 中，FSDP2 的应用逻辑分散在各个模型的 `parallelize.py` 文件中：

- `llama3/parallelize.py`: 274 行的 `apply_fsdp`，支持基本 FSDP2 功能
- `llama4/parallelize.py`: 182 行的 `apply_fsdp`，增加了 MoE 支持（shard_placement_fn、expert parallelism）
- `deepseek_v3/parallelize.py`: 引用 llama4 的 apply_fsdp
- `gpt_oss/parallelize.py`: 独立实现
- `flux/parallelize.py`: 独立实现

这种分散导致：
1. **代码重复**：每个模型都重复实现 `apply_fsdp`、`apply_replicate`、`disable_fsdp_gradient_division`
2. **维护困难**：修复 FSDP2 bug 需要在多个文件中同步修改
3. **功能不一致**：只有 llama4 完整支持 MoE FSDP2，其他模型无法使用
4. **新模型接入门槛高**：需要理解 FSDP2 内部机制才能正确实现

## Goals / Non-Goals

**Goals:**
- 将 FSDP2 应用逻辑提取到 `torchtitan/distributed/fsdp.py` 作为通用函数
- 支持标准 Transformer 和 MoE 模型的 FSDP2 应用
- 保持与现有模型配置的完全兼容
- 确保数值一致性（重构前后损失值相同）
- 降低新模型接入 FSDP2 的门槛

**Non-Goals:**
- 修改 FSDP2 内部实现（PyTorch 层面）
- 支持全新的并行策略
- 修改模型架构定义
- 优化训练性能（纯重构，无功能变更）

## Decisions

### Decision 1: 提取通用函数到 torchtitan/distributed/fsdp.py

**选择**：将 `apply_fsdp`、`apply_replicate`、`disable_fsdp_gradient_division` 移动到通用模块

**理由**：
- 遵循 CLAUDE.md 的 "Code Placement" 原则：模型无关代码应放在 `torchtitan/distributed/`
- 与 `ParallelDims`、`get_fsdp_reshard_after_forward_policy` 等现有 FSDP 工具集中管理

**替代方案**：创建 `torchtitan/models/common/parallelism.py`
- 拒绝理由：FSDP2 是分布式训练特性，应放在 distributed 包中

### Decision 2: 模型结构约定接口

**选择**：通过鸭子类型约定模型必须提供的属性：
```python
# 必需属性
model.layers: ModuleDict/Sequential  # Transformer blocks
model.tok_embeddings: Optional[nn.Module]  # 输入嵌入
model.norm: Optional[nn.Module]  # 最终归一化
model.output: Optional[nn.Module]  # 输出层

# MoE 相关（可选）
transformer_block.moe_enabled: bool  # 是否启用 MoE
transformer_block.moe: MoEContainer  # MoE 模块
```

**理由**：
- 无需修改现有模型类定义
- 与 torchtitan 现有模型结构一致
- 通过 `getattr` 提供默认值，保持向后兼容

**替代方案**：定义抽象基类 `FSDP2CompatibleModel`
- 拒绝理由：需要修改所有模型类，增加不必要的耦合

### Decision 3: MoE 支持策略

**选择**：提取 llama4 的 MoE FSDP2 逻辑为通用函数 `apply_fsdp_with_moe_support`

**理由**：
- llama4 已实现完整的 MoE FSDP2 支持（shard_placement_fn、EP/DP 混合 mesh）
- deepseek_v3 也是 MoE 模型，可复用此逻辑

**函数签名**：
```python
def apply_fsdp_to_model(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    # MoE 参数（可选）
    ep_degree: int = 1,
    edp_mesh: Optional[DeviceMesh] = None,
    gradient_divide_factor: Optional[int] = None,
) -> None:
```

### Decision 4: 渐进式重构策略

**选择**：保持现有 `parallelize.py` 函数签名不变，内部调用通用函数

**理由**：
- 最小化变更范围，降低风险
- 便于逐步验证每个模型
- 如果出现问题，容易回滚

**示例**：
```python
# llama3/parallelize.py
def apply_fsdp(model, dp_mesh, ...):
    from torchtitan.distributed.fsdp import apply_fsdp_to_model
    apply_fsdp_to_model(model, dp_mesh, ...)
```

## Risks / Trade-offs

**[Risk] 数值不一致** → **Mitigation**: 
- 使用 `--debug.seed=42 --debug.deterministic` 验证重构前后损失值
- 对每个模型运行集成测试

**[Risk] 模型特定逻辑丢失** → **Mitigation**:
- 仔细审计每个模型的 `apply_fsdp` 实现差异
- 保留模型特定的 hook 点

**[Risk] 新模型约束过紧** → **Mitigation**:
- 约定接口使用 `getattr` 提供默认值
- 文档中明确说明可选/必需属性

**[Risk] MoE 支持引入复杂度** → **Mitigation**:
- 非 MoE 模型无需提供 MoE 相关属性
- MoE 支持通过可选参数启用

## Migration Plan

1. **Phase 1**: 实现通用函数并添加到 `torchtitan/distributed/fsdp.py`
2. **Phase 2**: 重构 llama3（最简单的非 MoE 模型）并验证
3. **Phase 3**: 重构 llama4 并验证 MoE 支持
4. **Phase 4**: 重构 deepseek_v3、gpt_oss、flux
5. **Phase 5**: 添加单元测试，清理重复代码

**Rollback**: 如果发现问题，直接回滚到模型特定的 `parallelize.py` 实现。

## Open Questions

1. 是否需要支持自定义 FSDP2 应用顺序（某些模型可能需要先 FSDP 后 TP）？
2. `flux` 模型使用不同的架构（图像生成），其并行化逻辑是否兼容？
