<div align="center">

# FEMBA-Tiny 脑电基础模型量化与位宽临界点研究

在脑电基础模型 FEMBA-Tiny（双向 Mamba，867 万参数）上比较三种量化方案：训练后量化（PTQ）、FP32 预训练 + 量化微调（C）、量化预训练 + 量化微调（QAP）。下游任务覆盖 TUAR、TUAB、TUSL，本页汇总 INT8 和 INT4 的结果。

**TUAB 上 QAP 0.85、PTQ 0.76 · TUSL v2 上 QAP 0.86、PTQ 0.82 · INT8 存储 9.10 MB（3.8×）· INT4 存储 4.85 MB（7.1×）**

![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![PyTorch 2.7](https://img.shields.io/badge/pytorch-2.7.1%2Bcu128-ee4c2c)
![Brevitas 0.12.1](https://img.shields.io/badge/brevitas-0.12.1-008080)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-lightgrey.svg)](LICENSE)

[快速复现与分工](#5-快速复现与多机分工) · [INT8 完整结果](reproduction/formal_line/results/int8/README.md) · [INT4 完整结果](reproduction/formal_line/results/int4/README.md) · [评测指标与成本](reproduction/formal_line/results/metrics/README.md) · [PTQ 掉点定位](reproduction/formal_line/results/ptq-diagnosis/README.md) · [QAP 逐层分析](reproduction/formal_line/results/layers/README.md)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reproduction/figures/formal-overview-dark.svg">
  <img alt="TUAR、TUAB、TUSL v2 三任务 INT8 测试 AUROC（3 个随机种子均值与范围）：FP32 为 0.87、0.84、0.83，PTQ 为 0.86、0.76、0.82，C 为 0.86、0.84、0.83，QAP 为 0.86、0.85、0.86。" src="reproduction/figures/formal-overview-light.svg" width="780">
</picture>

</div>

本页的正式线实验采用受试者独立划分（预训练语料为 TUAB 官方训练集并剔除全部下游测试受试者），固定随机种子 42 / 43 / 44，使用 Brevitas 伪量化。

## 1. 正式线主结果：三任务 INT8（W8A8）对比

下表汇总三个随机种子的测试集均值（两位小数）：包含 ROC 曲线下面积（AUROC）、平衡准确率（BAcc，TUAR 与 TUAB 按默认阈值 `0`，TUSL v2 按最大概率分类）与期望校准误差（ECE，15 等宽分箱，越低越好）。`C` 为 FP32 预训练 + 量化微调，`QAP` 为量化预训练 + 量化微调，`随机初始化` 为不加载预训练直接做 INT8 量化微调。

| 任务 | 指标 | FP32 | PTQ | C | QAP | 随机初始化 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **TUAR** | AUROC ↑ | 0.87 | 0.86 | 0.86 | 0.86 | **0.87** |
| **TUAR** | 平衡准确率 BAcc ↑ | 0.75 | 0.63 | **0.76** | **0.76** | **0.76** |
| **TUAR** | 校准误差 ECE ↓ | 0.10 | 0.17 | 0.08 | 0.09 | **0.08** |
| **TUAB** | AUROC ↑ | 0.84 | 0.76 | 0.84 | **0.85** | **0.85** |
| **TUAB** | 平衡准确率 BAcc ↑ | 0.76 | 0.65 | 0.76 | **0.77** | **0.77** |
| **TUAB** | 校准误差 ECE ↓ | 0.03 | 0.06 | **0.02** | **0.02** | 0.03 |
| **TUSL v2** | 宏平均 AUROC ↑ | 0.83 | 0.82 | 0.83 | **0.86** | 0.82 |
| **TUSL v2（不计第 0 组）** | 宏平均 AUROC ↑ | **0.91** | 0.89 | 0.90 | 0.90 | 0.83 |
| **TUSL v2** | 宏平均 BAcc ↑ | 0.68 | 0.51 | 0.67 | **0.70** | 0.68 |
| **TUSL v2** | 校准误差 ECE ↓ | 0.11 | 0.16 | 0.12 | **0.05** | 0.12 |

若在验证集上重选分类阈值，PTQ 在 TUAR、TUAB 上的 BAcc 分别恢复到 `0.75` 与 `0.70`（其余各组变化不超过 `0.01`）。四位小数明细、逐种子标准差与配对差值见 [INT8 完整结果](reproduction/formal_line/results/int8/README.md) 与 [评测指标与成本](reproduction/formal_line/results/metrics/README.md)。

- **PTQ 在三个任务上掉点最重且校准最差**：TUAB 上 AUROC 比 FP32 低 `0.08`（`0.76` vs `0.84`）；TUAR 上默认阈值 BAcc 跌至 `0.63`（种子 42 把全部片段判为阳性，ECE 达 `0.33`），用验证集重选阈值后恢复到 `0.75`。
- **量化微调（C 与 QAP）能把精度拉回 FP32 水平**：三个任务的 AUROC 与 FP32 相差均不超过 `0.03`；TUAB 和 TUSL v2 上 QAP 在三个种子上都高于 PTQ（平均高 `+0.10` 和 `+0.04`）。
- **INT8 下量化预训练（QAP）与只做量化微调（C）持平**：TUAR 与 TUAB 上两组相差不超过 `0.01`，跨种子有正有负；TUSL v2 上 QAP 高出的 `+0.03` 全部来自第 0 测试组（单名受试者占 84% 事件，不计该组时两组均为 `0.90`）。
- **预训练在小样本 TUSL 上提升显著，在大样本 TUAB、TUAR 上主要加速收敛**：在每轮约 400 片段的 TUSL v2（不计第 0 组）上，预训练组（`0.90–0.91`）比随机初始化（`0.83`）高 `0.08`，三个种子与三个测试组同向；在每轮 12.3 万 / 5.5 万片段的 TUAB、TUAR 上与随机初始化持平，但预训练把验证损失见底从第 2–5 轮提前到第 1 轮。

## 2. 正式线主结果：三任务 INT4（W4A8）对比

权重降到 4 位，C 和 QAP 仍然不掉点。下表的口径和 INT8 相同。

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reproduction/figures/formal-overview-int4-dark.svg">
  <img alt="TUAR、TUAB、TUSL v2 三任务 INT4 测试 AUROC（3 个随机种子均值与范围）：FP32 为 0.87、0.84、0.83，PTQ 为 0.84、0.75、0.81，C 为 0.87、0.84、0.81，QAP 为 0.85、0.85、0.84。" src="reproduction/figures/formal-overview-int4-light.svg" width="780">
</picture>
</div>

| 任务 | 指标 | FP32 | PTQ | C | QAP | 随机初始化 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **TUAR** | AUROC ↑ | **0.87** | 0.84 | **0.87** | 0.85 | **0.87** |
| **TUAR** | 平衡准确率 BAcc ↑ | 0.75 | 0.57 | **0.76** | 0.75 | **0.76** |
| **TUAR** | 校准误差 ECE ↓ | 0.10 | 0.24 | 0.09 | 0.09 | **0.08** |
| **TUAB** | AUROC ↑ | 0.84 | 0.75 | 0.84 | **0.85** | **0.85** |
| **TUAB** | 平衡准确率 BAcc ↑ | 0.76 | 0.65 | **0.77** | **0.77** | **0.77** |
| **TUAB** | 校准误差 ECE ↓ | 0.03 | 0.09 | **0.02** | **0.02** | 0.03 |
| **TUSL v2** | 宏平均 AUROC ↑ | 0.83 | 0.81 | 0.81 | **0.84** | 0.82 |
| **TUSL v2（不计第 0 组）** | 宏平均 AUROC ↑ | **0.91** | 0.88 | 0.90 | 0.90 | 0.83 |
| **TUSL v2** | 宏平均 BAcc ↑ | **0.68** | 0.49 | 0.65 | **0.68** | **0.68** |
| **TUSL v2** | 校准误差 ECE ↓ | 0.11 | 0.18 | **0.09** | **0.09** | 0.11 |

四位小数明细、逐种子差值和 TUSL 重抽样见 [INT4 完整结果](reproduction/formal_line/results/int4/README.md)。

- **权重降到 4 位不掉点**：C、QAP 与 FP32 的 AUROC 均值差均不超过 `0.03`；相比 INT8，模型存储再省 `47%`（`9.10 → 4.85 MB`），量化层 BOPs 减半（`43.6 → 21.8 G`），合计 BOPs 再省 `25%`（`86.4 → 64.7 G`）。
- **PTQ 在 INT4 下继续下滑，必须靠量化微调或离群通道修复**：TUAB 上标准 W4A8 PTQ 比 FP32 低 `0.09`（`0.75` vs `0.84`），做 SmoothQuant 通道迁移后可不重训恢复到 `0.80`；TUAR 默认阈值下 BAcc 跌至 `0.57`（验证集重选阈值后恢复到 `0.74`）。
- **为什么 INT8 与 INT4 下只做量化微调（C）就够了，无需做量化预训练（QAP）**：位宽拆解表明 8 位与 4 位权重本身几乎不损精度（`W8A32` 为 `0.84`、`W4A32` 为 `0.83`），掉点主要来自第 2 个 Mamba 块的 8 位激活截断；下游量化微调（C）已将该处截断率从 `4%` 压到 `0.01%–0.02%`。因此 C 只需预训练一次 FP32 底座（`2.59 h`）即可通用于 INT8 与 INT4，而 QAP 每换一个位宽都要重训一次底座（约 `3.0 h`）却无稳定精度增益（INT4 下 QAP − C 在 TUAR 为 `−0.015`，TUAB 为 `+0.008`，TUSL 不计第 0 组为 `+0.003`）。
- **小样本任务仍需预训练底座**：与 INT8 一致，INT4 随机初始化在大样本 TUAB（`0.85`）、TUAR（`0.87`）上与预训练组持平，但在小样本 TUSL v2（不计第 0 组，每轮约 400 片段）上仅 `0.83`，比预训练组（`0.90`）低 `0.07`。

## 3. TUAB 上 PTQ 掉点定位与 QAP 逐层归因

TUAB 上 W8A8 PTQ 比 FP32 低 `0.08`（`0.84 → 0.76`）。分别关掉权重或激活量化、再逐层单独量化激活后，掉点拆解如下：

| TUAB 量化设置 | 权重位宽 | 激活位宽 | 测试 AUROC | 相对 FP32 |
| --- | --- | --- | ---: | ---: |
| **FP32 基线** | FP32 | FP32 | 0.84 | — |
| **只量化 8 位权重（W8A32）** | 8 bit | FP32 | 0.84 | 0.00 |
| **只量化 4 位权重（W4A32）** | 4 bit | FP32 | 0.83 | −0.01 |
| **只量化三值权重（Ternary）** | 2 bit | FP32 | 0.60 | −0.24 |
| **只量化 8 位激活（W32A8）** | FP32 | 8 bit | 0.76 | −0.08 |
| **标准 W8A8 PTQ（95 百分位）** | 8 bit | 8 bit | 0.76 | −0.08 |

逐层排名、长尾比值和不重训的修复实验见 [PTQ 掉点定位](reproduction/formal_line/results/ptq-diagnosis/README.md)，与 QAP 的逐块、逐处掉点对比见 [QAP 逐层分析](reproduction/formal_line/results/layers/README.md)。

- **8 位和 4 位掉在激活，三值掉在权重**：只量化 8 位权重不掉点（`0.84`），4 位权重仅多掉 `0.01`（`0.83`）；三值只量化权重就跌到 `0.60`，必须靠量化训练（C 或 QAP）恢复。
- **掉点集中在第 2 个 Mamba 块的扫描输出，而非离群值最大的第 1 块**：长尾最重的 7 处激活都在第 1 个 Mamba 块，单独量化时验证 AUROC 掉点不超过 `0.005`；真正引起掉点的是**第 2 个 Mamba 块的双向状态空间扫描（SSM）输出**。把这两处离群通道平移进下一层权重后，不重训的 W8A8 PTQ 可恢复到 `0.82`；QAP 相对 PTQ 减小的掉点也有 **76%** 来自第 2 个 Mamba 块。
- **INT4 的逐层结论相同**：QAP 减小的掉点有 **72%** 来自第 2 个 Mamba 块，差距最大的仍是反向扫描输出。

## 4. 位宽压缩率、推理计算量与训练耗时

FEMBA-Tiny 共 867 万参数：98.0% 为量化卷积与线性层权重，0.4%（3.1 万）为走 8 位激活量化的位置编码，其余 1.7%（depthwise 卷积、归一化、状态空间参数与偏置）保持 FP32。单次 5 秒脑电片段（`22 × 1280`）的比特操作数（BOPs）与单卡 RTX 5060 Ti（16 GB）预训练耗时如下：

| 位宽设置 | 权重×激活 | 有效位数 | 模型存储 | 压缩比 | 量化层 BOPs | 合计 BOPs | 预训练耗时 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| **FP32** | 32 × 32 | 32.00 bit | 34.68 MB | 1.0× | 696.9 G | 739.8 G | 2.59 h（独占） |
| **BF16 / FP16** | 16 × 16 | 16.00 bit | 17.34 MB | 2.0× | 174.2 G | 184.9 G | — |
| **INT8** | 8 × 8 | 8.40 bit | 9.10 MB | **3.8×** | 43.6 G | 86.4 G | 约 3.0 h（折算） |
| **INT4** | 4 × 8 | 4.48 bit | 4.85 MB | **7.1×** | 21.8 G | 64.7 G | 6.24 h（并发） |
| **三值** | 2 × 8 | 2.52 bit | 2.73 MB | **12.7×** | 10.9 G | 53.8 G | — |

`合计 BOPs` 在量化层之上计入了状态空间扫描与 depthwise 卷积（在 INT8 / INT4 / 三值下保持 FP32，共 `42.9 G`）。16 位未量化基线中，BF16 在三任务上零溢出且与 FP32 相差不超过 `0.0002`，FP16 在第 1 个 Mamba 块反向扫描输出有 `0.03%–0.11%` 的 TUAB 测试片段溢出。下游微调单卡耗时（TUAB `1.93–2.25 h`、TUAR `0.87–1.23 h`、TUSL 每组 `2–3 min`）见 [评测指标与成本](reproduction/formal_line/results/metrics/README.md)。

## 5. 快速复现与多机分工

正式线完整协议、环境变量与多机分跑步骤以 **[reproduction/formal_line/README.md](reproduction/formal_line/README.md)** 为准。

```bash
# 1. 安装正式线依赖（Python 3.11，PyTorch 2.7.1+cu128，Brevitas 0.12.1）
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r reproduction/formal_line/requirements-formal.txt

# 2. 检出上游模型源码（a2a4d44）并依次打入三个量化与三值补丁
git clone https://github.com/berumottovv00/BioFoundation.git "$SRC"
git -C "$SRC" checkout a2a4d441485efd8423c44b35d7448a30ccd44437
git -C "$SRC" apply "$PWD/reproduction/teammate_ptq/source.patch" \
                    "$PWD/reproduction/teammate_ptq/quantization-config.patch" \
                    "$PWD/reproduction/teammate_ptq/ternary-model.patch"

# 3. 运行单种子微调、PTQ 七档校准与评估队列（以 INT4 为例，复用共享 FP32 父模型）
FP32_PARENTS=given SEED=42 BITS=4 SCHEME=uniform \
  TASKS="tuar tuab tusl0-v2 tusl1-v2 tusl2-v2 tusl3-v2" LOG_SUFFIX=-all ./finetune_queue.sh
```

## 6. 报告问题与数据速查索引

| 报告问题 / 评测指标 | 结果页 | 原始 JSON 与复算脚本 |
| --- | --- | --- |
| **INT8 三任务主指标**（AUROC、AP、种子标准差、配对差值与重抽样） | [INT8 完整结果](reproduction/formal_line/results/int8/README.md) | [summary.json](reproduction/formal_line/results/int8/summary.json) · [provenance.json](reproduction/formal_line/results/int8/provenance.json) · [verify_binary.py](reproduction/formal_line/results/int8/verify_binary.py) |
| **INT4 三任务主指标**（AUROC、AP、配对差值、重抽样与运行环境） | [INT4 完整结果](reproduction/formal_line/results/int4/README.md) | [summary.json](reproduction/formal_line/results/int4/summary.json) · [provenance.json](reproduction/formal_line/results/int4/provenance.json) · [tusl-bootstrap.json](reproduction/formal_line/results/int4/tusl-bootstrap.json) |
| **平衡准确率（BAcc）**（默认阈值 `0` vs 验证集阈值，及 TUSL 最大概率） | [评测指标与成本 §平衡准确率](reproduction/formal_line/results/metrics/README.md#平衡准确率) | [int8/summary.json](reproduction/formal_line/results/int8/summary.json) · [int4/summary.json](reproduction/formal_line/results/int4/summary.json) |
| **期望校准误差（ECE）**（15 等宽分箱，含 TUSL 剔除第 0 组对照） | [评测指标与成本 §校准误差](reproduction/formal_line/results/metrics/README.md#校准误差) | [calibration-error.json](reproduction/formal_line/results/metrics/calibration-error.json) · [calibration_error.py](reproduction/formal_line/calibration_error.py) |
| **16 位未量化基线**（BF16 与 FP16 三任务重放及 FP16 溢出位置统计） | [评测指标与成本 §FP16 和 BF16](reproduction/formal_line/results/metrics/README.md#fp16-和-bf16) | [half-precision.json](reproduction/formal_line/results/metrics/half-precision.json) · [half_precision.py](reproduction/teammate_ptq/half_precision.py) |
| **随机初始化基线**（不加载预训练直接做 INT8、INT4 量化微调，检验预训练收益） | [评测指标与成本 §随机初始化](reproduction/formal_line/results/metrics/README.md#随机初始化) | [random-init.json](reproduction/formal_line/results/metrics/random-init.json) · [run_random_init.sh](reproduction/formal_line/run_random_init.sh) |
| **模型成本与训练耗时**（有效位数、存储、量化层与合计 BOPs、预训练与微调耗时） | [评测指标与成本 §每参数有效位数](reproduction/formal_line/results/metrics/README.md#每参数有效位数和存储) | [model-cost.json](reproduction/formal_line/results/metrics/model-cost.json) · [model_cost.py](reproduction/formal_line/model_cost.py) |
| **QAP 在哪一层减小掉点最多？**（TUAB INT8、INT4 按块 + 第 2 块逐处 + SQNR/截断率） | [QAP 逐层分析](reproduction/formal_line/results/layers/README.md) | [layers/seed*.json](reproduction/formal_line/results/layers/) · [layer_comparison.py](reproduction/teammate_ptq/layer_comparison.py) |
| **PTQ 掉点诊断与修复**（位宽拆解、MSE/KL 校准、FP16 回退、SmoothQuant 迁移、Hadamard 旋转） | [PTQ 掉点定位](reproduction/formal_line/results/ptq-diagnosis/README.md) | [ptq-diagnosis/summarize.py](reproduction/formal_line/results/ptq-diagnosis/summarize.py) · [channel_migration.py](reproduction/teammate_ptq/channel_migration.py) · [rotation_ptq.py](reproduction/teammate_ptq/rotation_ptq.py) |
| **正式线协议、数据划分与多机队列**（[subject_split](reproduction/subject_split/)、[tusl_events](reproduction/tusl_events/)、[teammate_ptq](reproduction/teammate_ptq/README.md)） | [正式线复现手册](reproduction/formal_line/README.md) · [早期 2025 权重归档](reproduction/RESULTS.md) | [finetune_queue.sh](reproduction/formal_line/finetune_queue.sh) · [summarize_formal.py](reproduction/formal_line/summarize_formal.py) |

## 上游来源与致谢

- **量化模型与 QAP 基线**：基于 [berumottovv00/BioFoundation](https://github.com/berumottovv00/BioFoundation/tree/a2a4d441485efd8423c44b35d7448a30ccd44437)（`a2a4d44`）。
- **FEMBA 原始论文与公开实现**：[pulp-bio/BioFoundation](https://github.com/pulp-bio/BioFoundation/tree/d88596590f3bd3fce573be07646b7d3977ce7bcc) · [FEMBA（EMBC 2025）](https://arxiv.org/abs/2502.06438v2) · [FEMBA on the Edge（IEEE TBME 2026）](https://arxiv.org/abs/2603.26716v1) · [Apache-2.0 License](LICENSE)。
