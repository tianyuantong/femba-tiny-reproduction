# 复现步骤与实验口径

## 两条 FP32 结果线

公开代码轨迹使用 18 类、占比 ≥30% 的标签和 AdamW 5e-4，AUROC 为 0.923889。
论文标签轨迹使用 13 类、任一出现的标签：5e-4 组为 0.913040；已完成的 1e-4 对照
为 0.898821，并按最低验证 loss 选作精度母模型。论文标签组的归一化、优化器类型、
模型及其余训练设置沿用公开代码。学习率探索已结束，后续精度实验固定该母模型。

## 环境与输入

从仓库根目录运行。参考环境：WSL、RTX 5060 Ti、Python 3.11、PyTorch 2.7.1+cu128、
Lightning 2.6.5、TorchMetrics 1.9.0、mamba-ssm 2.3.2.post1。
当前环境有 selective-scan CUDA 扩展，未安装可选 causal-conv1d 扩展。

训练适配器读取官方提交 `d88596590f3bd3fce573be07646b7d3977ce7bcc` 的模型、task、
loader、归一化和 scheduler。先确认本地 Git 对象存在，并设置已有文件路径：

```bash
git cat-file -e d88596590f3bd3fce573be07646b7d3977ce7bcc^{commit}
export FEMBA_WEIGHT=/path/to/FEMBA_tiny.safetensors
export PAPER_ROOT=/path/to/existing-paper-label-run
export LR_ROOT=/path/to/new-lr-run
export PRECISION_ROOT=/path/to/new-precision-run
```

`PAPER_ROOT` 应包含 `results/upstream2025-data.json` 和
`data/TUAR_data/upstream2025_{train,val,test}.h5`。标签 H5 通过相对 ExternalLink
引用原始信号，需同时保留其源 H5。`LR_ROOT`、`PRECISION_ROOT` 使用新目录。

若已有经过核验的论文标签数据，直接继续训练；尚未生成时运行一次：

```bash
python scripts/femba_paper_labels_prepare.py \
  --parent-manifest "$PARENT_MANIFEST" --processed "$PROCESSED_ROOT" \
  --output-root "$PAPER_ROOT"
```

`PARENT_MANIFEST` 指向已有公开代码数据清单，`PROCESSED_ROOT` 指向对应预处理 PKL。
脚本先核对原标签与信号身份，再生成论文标签和映射证据。

## 已完成 FP32 训练的复现入口

```bash
FEMBA_RUN_ROOT="$PAPER_ROOT" python scripts/femba_upstream_2025_train.py probe \
  --output-root "$LR_ROOT" --learning-rate 1e-4
FEMBA_RUN_ROOT="$PAPER_ROOT" python scripts/femba_upstream_2025_train.py train \
  --output-root "$LR_ROOT" --learning-rate 1e-4
FEMBA_RUN_ROOT="$LR_ROOT" python scripts/femba_upstream_2025_audit.py
```

| 设置 | 固定值 |
| --- | --- |
| 模型 | Tiny，embed_dim=35，2 个双向 Mamba 块，原版分类头 |
| 初始化 | 发布的 TUAR Tiny 编码器，43 个张量逐项一致加载 |
| 随机种子与精度 | seed 42，`32-true`，matmul／cuDNN TF32 均关闭 |
| 优化器 | AdamW，betas=(0.9,0.999)，weight_decay=0.5，layer decay=0.75 |
| 调度 | cosine，30 轮，预热 10 轮，初始预热 LR=2.5e-7，最低 LR=2.5e-6 |
| 全局批次 | 单卡 batch 256 × 累积 4，每轮 216 个微批次／54 次更新 |
| 更新预算 | 共 1,620 步，预热 540 步；每轮舍去末尾 876 个训练窗口 |

已完成的学习率对照分别使用 5e-4 和 1e-4，并写入不同输出目录。每次训练取最低验证
交叉熵检查点，再比较两组最低验证损失；完全相同时保留原 5e-4 组。
本次选中 1e-4、第 7 轮／378 步，检查点 SHA256 为
`233b15df818d93b964a3d5c305a184de2ef1e62c2d50119a4349fb13b30c8890`。
测试分数只用于报告，final 检查点用于观察训练后期变化。

审计从运行时保存的 config 重建模型，检查完整测试预测与指标；配置、执行源码
快照和输出均由新目录保存，原数据与旧结果保持独立。

## 固定数据和指标

处理采用 0.1–75 Hz 带通、60 Hz 陷波、256 Hz 采样、22 通道双极导联、5 秒不重叠窗口。
seed 42 记录级划分得到 56,172／5,899／7,090 个训练／验证／测试窗口。
标签为 13 类伪迹任一出现，沿用 256 Hz 网格上的 Python round（恰好半数取偶）和半开区间。
训练／验证／测试正类数为 28,746／3,374／4,094。

输入在 FP32 中按原版固定上下界公式 `(x+20)/(40+1e-8)` 归一化。
论文采用 IQR 归一化和 Adam；本轮保留公开代码的归一化及 AdamW。
实际受试者交叉数为 train–val 10、train–test 11、val–test 3，作者划分名单未公开。

验证和测试保留全部样本，batch_size=256。主指标逐批复用原版 TorchMetrics：
分类标签取 `softmax(logits).argmax`，AUROC／AP 输入正类原始 logit，保留其逐批
自动 sigmoid 行为。所有版本的评分张量转为 FP32；全局 softmax 指标单独保存为诊断值。

## 精度实验

将审计后冻结的母模型目录和 SHA256 填入变量，再执行完整矩阵：

```bash
export SELECTED_PARENT_ROOT=/path/to/selected-audited-run
export SELECTED_CHECKPOINT_SHA256=233b15df818d93b964a3d5c305a184de2ef1e62c2d50119a4349fb13b30c8890
python scripts/femba_precision_suite.py \
  --parent-root "$SELECTED_PARENT_ROOT" --output-root "$PRECISION_ROOT" \
  --checkpoint-sha256 "$SELECTED_CHECKPOINT_SHA256" \
  --rotation-parity vector-rms-2026-09-22
```

以上哈希对应已报告的母模型；重新训练时使用新审计输出的实际哈希。
每一行严格加载同一母模型，再应用自己的精度配置。

单独复现 22 个 Linear 输入边界组时，另用新目录：

```bash
export LINEAR_PRECISION_ROOT=/path/to/new-linear-input-run
python scripts/femba_precision_suite.py \
  --parent-root "$SELECTED_PARENT_ROOT" --output-root "$LINEAR_PRECISION_ROOT" \
  --checkpoint-sha256 "$SELECTED_CHECKPOINT_SHA256" \
  --variants fp32 w8a8-linear w4a8-linear
```

首行重复 FP32，核对其完整 logits 和指标与冻结母模型一致，再比较两行 A8。
两种激活边界分别采集未量化 FP32 统计，均使用同一校准集合；各组内部 W8／W4 共用统计。

单独运行修订协议下的四个旋转版本，使用新的 `ROTATION_ROOT`：

```bash
export ROTATION_ROOT=/path/to/new-rotation-rms-run
python scripts/femba_precision_suite.py \
  --parent-root "$SELECTED_PARENT_ROOT" --output-root "$ROTATION_ROOT" \
  --checkpoint-sha256 "$SELECTED_CHECKPOINT_SHA256" \
  --rotation-parity vector-rms-2026-09-22 \
  --variants fp32 rot-w8a32 rot-w4a32 rot-w8a8-float rot-w4a8-float
```

FP32 同样作为重复锚点。Rot-A8 沿用原 57 处量化位置，与现有 57 处未旋转组比较；
校准集合和尺度规则保持不变。省略 `--rotation-parity` 时保留原逐元素协议。

- **权重：** 28 个 Linear／Conv 张量，逐输出通道对称量化。W8 范围 [-127,127]，
  W4 范围 [-7,7]，round→clamp→dequantize，偏置保留浮点。
- **57 处激活组：** 57 个固定位置，A8 逐张量对称量化。校准取 seed 42 固定的 2,048 个训练窗口，
  每类 1,024 个；关闭 W/A 量化，采集一次 FP32 统计，W8／W4 及两种尺度共用。
  `s_float=max_abs/127`，全零时取 1；`s_pot=2**ceil(log2(s_float))`。评估期间尺度冻结。
- **22 个 Linear 输入边界组：** 五个 Mamba 的 `in_proj`、`x_proj`、`dt_proj`、
  `out_proj` 输入，加上分类头 `fc1`、`fc3` 输入，共 22 处。固定同一母模型、
  权重量化、校准集合和尺度规则，仅改变激活量化位置。`x_proj` 使用独立的量化
  临时输入，保留传入 scan 的原始 x；不直接量化 scan 的 x／B／C／delta。
- **FP16：** direct 直接转换；state-safe 将 `A_log`、`D`、`dt_proj.bias` 保留 FP32；
  AMP 使用 FP16 autocast。检测到非有限输出时保存失败位置并停止该版本。
- **旋转：** 五个 Mamba 输出投影采用 seed 42 固定的置换／符号／H128 变换，先变换再量化。
  使用单独的旋转 FP32 校准统计，Rot-A8 保留 57 处激活边界。数值门槛按下述修订协议验收。
  固定 h/W 的 FP64 隔离对照仅用于检查变换公式；正式推理、校准和评分保持 FP32，
  详见[数值诊断](rotation_numeric_results.json)。

QDQ 运算在 FP32 中模拟，selective scan、状态参数、归一化和非线性保留浮点。
W4A8 表示 4 位权重、8 位激活；本轮测量精度变化。

## 核验

32 个训练和 32 个验证探针用于检查原生／显式图：五个 Mamba 输出投影和 logits
逐元素满足 `rtol=1e-4, atol=1e-5`。A8 同时依赖未量化权重和对应 W8／W4 路径的检查。

### 2026-09-22：旋转数值检查修订

固定 h/W 的诊断中，10 组 FP64 旋转结果均与参考吻合，而原生 FP32 自身相对
FP64 参考也有逐元素超限。大数抵消使接近零的单个分量对运算顺序敏感，故将旋转
中间张量的检查改为逐特征向量的误差范数：对五个输出投影的每个 `[B,T]` 位置，要求

```text
RMS(rotated - reference) <= 1e-5 + 1e-4 * RMS(reference)
RMS(v) = sqrt(mean(v ** 2))  # mean over the feature dimension only
```

每个特征向量都须满足门槛；最终 logits 仍使用原 `rtol=1e-4, atol=1e-5` 逐元素检查。
旋转基、seed 和探针样本固定，identity、H128 正对照必须通过；完整单位基结构检查
须与固定的置换／符号／分块 Hadamard 一致。H4-only 负对照分别覆盖四个 1540 维目标：
每次只把一个目标末尾的 H4 在激活和权重两侧同时换成 I4，保持置换、符号和 12 个 H128 块。
即使输出仍线性等价，也必须被结构检查拒绝。激活单侧转置、双侧交换置换／符号顺序
负对照同样必须失败。旧逐元素检查和 FP64 诊断保留在结果中。

通过数值检查后，四个 Rot 版本须完成全部验证／测试窗口、有限性检查、保存重载和
指标重放，精度提高或退化都如实记录。旋转用于检验激活幅值重分布对静态 PTQ 的影响，
为后续 QAT 提供输入；本轮不进行 QAT 训练。

完成评估的版本验证模型与尺度保存／重载后探针 logits 逐位一致，独立按原 batch
重放指标，并核对完整预测顺序、标签、输入、母模型和校准集合哈希。
原生 Accuracy、Balanced Accuracy、AUROC、AP 任一项下降超过 0.005 标为精度退化；
数值失败与一致性失败单独记录。结果见[RESULTS.md](RESULTS.md)。

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```
