## ADDED Requirements

### Requirement: 通用 FSDP2 应用函数
系统 SHALL 提供 `apply_fsdp_to_model` 函数，支持对任意符合约定的模型应用 FSDP2。

#### Scenario: 标准 Transformer 模型应用 FSDP2
- **WHEN** 调用 `apply_fsdp_to_model(model, dp_mesh, param_dtype, reduce_dtype, pp_enabled=False)`
- **THEN** 模型被 FSDP2 包装，支持数据并行训练

#### Scenario: MoE 模型应用 FSDP2
- **WHEN** 调用 `apply_fsdp_to_model(model, dp_mesh, ..., ep_degree=4, edp_mesh=mesh)`
- **THEN** MoE 专家的参数使用 edp_mesh 分片，其他参数使用 dp_mesh 分片

#### Scenario: CPU Offload 支持
- **WHEN** 调用 `apply_fsdp_to_model(model, ..., cpu_offload=True)`
- **THEN** 模型参数被卸载到 CPU，仅在需要时加载到 GPU

### Requirement: 模型结构约定
系统 SHALL 通过鸭子类型定义 FSDP2 应用所需的模型属性，支持必需和可选属性。

#### Scenario: 必需属性存在
- **GIVEN** 模型具有 `layers`、`tok_embeddings`、`norm`、`output` 属性
- **WHEN** 调用 `apply_fsdp_to_model`
- **THEN** FSDP2 成功应用

#### Scenario: 可选属性缺失
- **GIVEN** 模型缺少 `tok_embeddings`（如 `None`）
- **WHEN** 调用 `apply_fsdp_to_model`
- **THEN** FSDP2 跳过该模块，继续应用其他模块

#### Scenario: MoE 属性检测
- **GIVEN** 模型的 `layers` 中的模块具有 `moe_enabled` 和 `moe` 属性
- **WHEN** 调用 `apply_fsdp_to_model` 且 `ep_degree > 1`
- **THEN** 对 MoE 模块应用专家并行特定的 FSDP2 配置

### Requirement: HSDP 支持
系统 SHALL 提供 `apply_replicate_to_model` 函数，支持纯数据并行（无分片）场景。

#### Scenario: 纯数据并行训练
- **WHEN** 调用 `apply_replicate_to_model(model, dp_mesh, param_dtype, reduce_dtype)`
- **THEN** 模型参数被复制到所有 DP rank，使用 `replicate` 而非 `fully_shard`

### Requirement: 梯度除法控制
系统 SHALL 提供 `disable_fsdp_gradient_division` 函数，禁用 FSDP 自动梯度除法。

#### Scenario: 禁用梯度除法
- **GIVEN** 模型已应用 FSDP2
- **WHEN** 调用 `disable_fsdp_gradient_division(model)`
- **THEN** 所有 FSDP 模块的 `gradient_divide_factor` 被设置为 1.0

### Requirement: 现有模型兼容性
系统 SHALL 保持与所有现有模型的兼容性，重构后训练行为不变。

#### Scenario: llama3 模型训练
- **GIVEN** llama3 8B 模型配置
- **WHEN** 运行训练脚本 `NGPU=8 ./run_train.sh`
- **THEN** 损失值与重构前完全一致（使用 `--debug.seed=42 --debug.deterministic`）

#### Scenario: llama4 MoE 模型训练
- **GIVEN** llama4 MoE 模型配置，启用 EP=4
- **WHEN** 运行训练脚本
- **THEN** 专家参数正确分片，训练正常进行

#### Scenario: deepseek_v3 模型训练
- **GIVEN** deepseek_v3 模型配置
- **WHEN** 运行训练脚本
- **THEN** 复用 llama4 的 MoE FSDP2 逻辑，训练正常进行
