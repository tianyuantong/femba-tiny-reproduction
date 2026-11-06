# 历史实验归档：2025 公开权重 TUAR 复现步骤

> **正式线（按受试者划分，覆盖 TUAR / TUAB / TUSL 三任务与 INT8 / INT4 / 三值）见 [reproduction/formal_line/](formal_line/README.md)。** 本页与 [RESULTS.md](RESULTS.md) 归档最早基于 2025 公开代码（`d885965`）与 `mamba-ssm` 的 TUAR 单种子（seed 42，按录音划分）复现步骤。

本实验采用论文的 13 类伪迹定义（5 秒片段内任一伪迹出现即为正类）；输入归一化、AdamW 优化器和模型结构沿用 2025 公开代码。FP32 基线模型按最低验证交叉熵损失选择，后续各量化与旋转配置共用同一个 FP32 检查点。

## 环境与输入

从仓库根目录运行。环境：WSL、RTX 5060 Ti、Python 3.11、PyTorch 2.7.1+cu128、Lightning 2.6.5、TorchMetrics 1.9.0、`mamba-ssm` 2.3.2.post1（含 `selective-scan` CUDA 扩展）。

训练脚本直接复用官方提交 `d88596590f3bd3fce573be07646b7d3977ce7bcc` 的模型、数据加载器、归一化和学习率调度器。运行前先确认本地存在该提交，并设置路径变量：

```bash
git cat-file -e d88596590f3bd3fce573be07646b7d3977ce7bcc^{commit}
export FEMBA_WEIGHT=/path/to/FEMBA_tiny.safetensors
export PAPER_ROOT=/path/to/existing-paper-label-run
export LR_ROOT=/path/to/new-lr-run
export PRECISION_ROOT=/path/to/new-precision-run
```

`PAPER_ROOT` 包含 `results/upstream2025-data.json` 和 `data/TUAR_data/upstream2025_{train,val,test}.h5`（通过 HDF5 ExternalLink 指向原始信号文件）。如尚未生成论文 13 类标签数据，先运行一次转换脚本：

```bash
python scripts/femba_paper_labels_prepare.py \
  --parent-manifest "$PARENT_MANIFEST" --processed "$PROCESSED_ROOT" \
  --output-root "$PAPER_ROOT"
```

## FP32 基线训练

```bash
FEMBA_RUN_ROOT="$PAPER_ROOT" python scripts/femba_upstream_2025_train.py probe \
  --output-root "$LR_ROOT" --learning-rate 1e-4
FEMBA_RUN_ROOT="$PAPER_ROOT" python scripts/femba_upstream_2025_train.py train \
  --output-root "$LR_ROOT" --learning-rate 1e-4
FEMBA_RUN_ROOT="$LR_ROOT" python scripts/femba_upstream_2025_audit.py
```

| 设置 | 数值 |
| --- | --- |
| 模型结构 | FEMBA-Tiny，`embed_dim=35`，2 个双向 Mamba 块，原版分类头 |
| 预训练初始化 | 官方发布的 `FEMBA_tiny.safetensors`（43 个编码器张量） |
| 随机种子与精度 | seed 42，FP32（关闭 matmul 与 cuDNN 的 TF32） |
| 优化器 | AdamW，`betas=(0.9, 0.999)`，`weight_decay=0.5`，逐层衰减 `0.75` |
| 学习率调度 | 余弦退火 30 轮，预热 10 轮，初始学习率 `2.5e-7`，最低学习率 `2.5e-6` |
| 全局批次与步数 | 单卡 batch 256 × 梯度累积 4，每轮 54 步，共 1,620 步（每轮末尾不足一整批的 876 个训练片段丢弃） |

对比 `5e-4` 和 `1e-4` 两个学习率后，按最低验证交叉熵选出 `1e-4` 组第 7 轮（378 步）的检查点作为后续量化实验的共用 FP32 基线（SHA256：`233b15df818d93b964a3d5c305a184de2ef1e62c2d50119a4349fb13b30c8890`）。

## 数据与指标口径

- **预处理与划分**：0.1–75 Hz 带通、60 Hz 陷波、256 Hz 采样、22 通道双极导联、5 秒不重叠片段。seed 42 按录音划分得到训练 56,172、验证 5,899、测试 7,090 个片段（正类数分别为 28,746、3,374、4,094；训练与测试之间有 11 名受试者重叠）。
- **归一化**：按公开代码的固定上下界公式 `(x + 20) / (40 + 1e-8)` 归一化。
- **评估指标**：沿用公开代码的 TorchMetrics 计算方式——分类预测取 `softmax(logits).argmax`，AUROC 与平均精度（AP）传入正类原始 logit `z1`。

## 量化与旋转实验

将选定的 FP32 基线检查点目录与哈希填入环境变量，运行各量化配置：

```bash
export SELECTED_PARENT_ROOT=/path/to/selected-audited-run
export SELECTED_CHECKPOINT_SHA256=233b15df818d93b964a3d5c305a184de2ef1e62c2d50119a4349fb13b30c8890
python scripts/femba_precision_suite.py \
  --parent-root "$SELECTED_PARENT_ROOT" --output-root "$PRECISION_ROOT" \
  --checkpoint-sha256 "$SELECTED_CHECKPOINT_SHA256" \
  --rotation-parity vector-rms-2026-09-22
```

单独运行“只量化 22 个线性层输入”的对照：

```bash
export LINEAR_PRECISION_ROOT=/path/to/new-linear-input-run
python scripts/femba_precision_suite.py \
  --parent-root "$SELECTED_PARENT_ROOT" --output-root "$LINEAR_PRECISION_ROOT" \
  --checkpoint-sha256 "$SELECTED_CHECKPOINT_SHA256" \
  --variants fp32 w8a8-linear w4a8-linear
```

单独运行四个 Hadamard 旋转（H128）配置：

```bash
export ROTATION_ROOT=/path/to/new-rotation-rms-run
python scripts/femba_precision_suite.py \
  --parent-root "$SELECTED_PARENT_ROOT" --output-root "$ROTATION_ROOT" \
  --checkpoint-sha256 "$SELECTED_CHECKPOINT_SHA256" \
  --rotation-parity vector-rms-2026-09-22 \
  --variants fp32 rot-w8a32 rot-w4a32 rot-w8a8-float rot-w4a8-float
```

各配置的具体规则：

- **权重量化**：覆盖 28 个线性层与卷积层权重，逐输出通道对称量化（W8 范围 `[-127, 127]`，W4 范围 `[-7, 7]`，偏置保持 FP32）。
- **57 处激活量化**：逐张量对称量化。校准集固定取训练集中 2,048 个片段（正负类各 1,024 个），先关掉量化统计一次 FP32 最大绝对值 `max_abs`。浮点步长 `s_float = max_abs / 127`，2 的幂次步长 `s_pot = 2 ** ceil(log2(s_float))`。
- **22 处线性层输入激活量化**：只量化五个 Mamba 块的 `in_proj`、`x_proj`、`dt_proj`、`out_proj` 输入与分类头 `fc1`、`fc3` 输入，不再直接量化状态空间扫描内部的 `x / B / C / delta`。
- **FP16 三种路径**：`FP16-direct`（全模型转 FP16）、`FP16-state-safe`（保持 `A_log`、`D`、`dt_proj.bias` 为 FP32）、`AMP-FP16`（PyTorch 自动混合精度）。一旦检测到 `NaN` 或 `Inf`，记录首次出现的位置并停止该配置。
- **Hadamard 旋转（H128）**：在五个 Mamba 输出投影的激活与权重两侧配对乘上正交变换矩阵（随机置换 + 随机正负号 + 分块 128 维 Hadamard 矩阵），压低激活峰值后再做量化（数值核对见 [rotation_numeric_results.json](rotation_numeric_results.json)）。

量化通过 FP32 里的量化-反量化（QDQ）模拟，状态空间扫描、归一化与激活函数保持浮点。

## 数值核对

### 2026-09-22：旋转数值检查修订

未量化时，激活与权重两侧配对正交旋转在数学上不改变输出，但 FP32 浮点加法的重排会在接近 0 的单个元素上产生大数相消误差。因此，五个输出投影的等价性检查按每个时间步的特征向量均方根（RMS）误差验收：

```text
RMS(rotated - reference) <= 1e-5 + 1e-4 * RMS(reference)
RMS(v) = sqrt(mean(v ** 2))  # mean over the feature dimension only
```

最终分类 logits 仍按逐元素 `rtol=1e-4, atol=1e-5` 核对。单元测试同时包含单位阵与合法 H128 的正向验证，以及单侧漏转置、局部替换为单位块等 6 项反向拦截测试（详见 [RESULTS.md](RESULTS.md)）：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```
