# 困难约束纠偏实验 v2.1 执行说明

## 研究边界

- 旧 `frozen_test_v1`、35 个旧检查点及 19,600 条旧结果原样保留，只作为宽松工况证据。
- 本协议先证明任务本身是“完整覆盖不可行、部分覆盖可行”，再评价或训练模型。
- 场景筛选阶段不得运行 PPO、MLP、A2C、贪心或元启发式算法。
- 正式测试场景必须等训练分支冻结后才生成；测试开始后不得修改模型、场景或判优规则。

## 当前协议身份

- 协议：`paper_runs/protocols/difficulty_test_v2_1/protocol.json`
- 协议哈希：`faf5d8be36652a26d7452c6ed128d9cac11775002683297e0a9da1120db799b4`
- 父困难协议：`difficulty_test_v2`
- 父宽松协议：`frozen_test_v1`

## 分阶段命令

以下命令均在项目根目录运行，并使用 `Deeplearning-gpu` 环境。

### 1. 生成并认证困难 validation

```powershell
python -B paper_difficulty_experiments.py prepare --split validation --resume-existing
python -B paper_difficulty_experiments.py audit-environment --manifest paper_runs/difficulty_v2_1/manifests/validation
```

验收：108 个场景，三个难度层、三种节点规模、四种约束类型和三种优先级布局完整；每个场景均有完整覆盖不可行证书和安全非空返航路线。

### 2. 生成并认证困难训练池

```powershell
python -B paper_difficulty_experiments.py prepare --split training_pool --resume-existing
python -B paper_difficulty_experiments.py audit-environment --manifest paper_runs/difficulty_v2_1/manifests/training_pool
```

验收：648 个场景，且与 validation 场景内容无重叠。

### 3. 旧检查点资格验证

```powershell
python -B paper_difficulty_experiments.py qualify-existing --manifest paper_runs/difficulty_v2_1/manifests/validation --device cuda --resume-existing
```

门禁只允许两种结论：

- `keep_all_35`：35 个检查点全部保留；
- `retrain_all_35`：35 个检查点全部重训。

不得部分保留、部分重训。`no_return_reserve` 只要求结果完整可审计，不使用 95% 安全率门槛。

### 4. 仅在 `retrain_all_35` 时执行 1,800 回合试训

```powershell
python -B paper_difficulty_experiments.py train-grid --stage pilot --training-manifest paper_runs/difficulty_v2_1/manifests/training_pool --validation-manifest paper_runs/difficulty_v2_1/manifests/validation --device cuda --resume-existing
python -B paper_difficulty_experiments.py assess-pilot --validation-manifest paper_runs/difficulty_v2_1/manifests/validation --device cuda
```

固定检查点为 100、200、400、600 回合。试训模型只用于环境和训练健康检查，永不进入论文正式比较。

只允许因以下原因停止并修订：

- 数据或实现错误；
- 三个核心模型与两个固定贪心共同触及覆盖天花板；
- 三个核心模型共同过难或共同策略崩溃；
- 大量认证状态与统一评价器矛盾。

只有 PPO＋Pointer 暂时落后时，不得调整任务。

### 5. 正式训练与训练分支冻结

仅在试训通过时运行：

```powershell
python -B paper_difficulty_experiments.py train-grid --stage formal --training-manifest paper_runs/difficulty_v2_1/manifests/training_pool --validation-manifest paper_runs/difficulty_v2_1/manifests/validation --device cuda --resume-existing
python -B paper_difficulty_experiments.py freeze-training --training-manifest paper_runs/difficulty_v2_1/manifests/training_pool --validation-manifest paper_runs/difficulty_v2_1/manifests/validation
```

正式训练为 7 变体 × 5 种子 × 3,000 回合。固定健康检查点为 250、500、1000、1500、2000、2500、3000 回合。

### 6. 训练冻结后生成正式 test

```powershell
python -B paper_difficulty_experiments.py prepare-formal-test --training-freeze paper_runs/difficulty_v2_1/training_freeze.json --resume-existing
python -B paper_difficulty_experiments.py audit-environment --manifest paper_runs/difficulty_v2_1/manifests/formal_test
```

验收：216 个独立正式场景，与训练池和 validation 均无重叠。

### 7. 正式评价与判优

```powershell
python -B paper_difficulty_experiments.py evaluate-formal --formal-manifest paper_runs/difficulty_v2_1/manifests/formal_test --training-freeze paper_runs/difficulty_v2_1/training_freeze.json --family all --device cuda --resume-existing
python -B paper_difficulty_experiments.py analyze-formal
```

正式结果固定为：

- 学习模型：7,560 条；
- 主传统基线：7,128 条；
- 补充算法：2,592 条；
- 合计：17,280 条。

唯一确认性主指标为安全加权覆盖率。分析程序只有在预注册的覆盖率、显著性、效应方向、五种子一致性、安全非劣效和 Pareto 条件全部通过时，才允许输出“最佳学习算法”措辞。

## 恢复和故障处理

- 长任务只使用相同参数加 `--resume-existing` 恢复，不得重复启动。
- 任一协议、代码、manifest、训练池或检查点身份漂移，立即停止。
- 修改训练分布、奖励、掩码或网络后，当前正式训练全部作废，新建协议版本并从 0 回合重训。
- 任何失败、不安全、零访问或未返航结果都必须保留。
