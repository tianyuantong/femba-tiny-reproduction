<div align="center">

# FEMBA-Tiny 脑电基础模型量化与位宽临界点研究

在脑电基础模型 FEMBA-Tiny（双向 Mamba，867 万参数）上比较三种量化方案：训练后量化（PTQ，post-training quantization）、FP32 预训练 + 量化感知微调（QAT，quantization-aware training）、量化预训练 + 量化微调（QAP，quantization-aware pretraining）。下游任务来自 TUH 脑电数据库：TUAR 伪迹检测、TUAB 异常检测、TUSL 慢波三分类。本页汇总 INT8 和 INT4；三值实验由队友负责，不在本仓库。

**TUAB 测试 AUROC：PTQ 0.76，QAT 0.84，QAP 0.85 · INT4 存储 4.85 MB（7.1×），精度不降 · QAP 与 QAT 持平**

![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![PyTorch 2.7](https://img.shields.io/badge/pytorch-2.7.1%2Bcu128-ee4c2c)
![Brevitas 0.12.1](https://img.shields.io/badge/brevitas-0.12.1-008080)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-lightgrey.svg)](LICENSE)

[问题与数据速查](#6-报告问题与数据速查索引) · [INT8 结果](reproduction/formal_line/results/int8/README.md) · [INT4 结果](reproduction/formal_line/results/int4/README.md) · [评测指标与成本](reproduction/formal_line/results/metrics/README.md) · [QAP 逐层分析](reproduction/formal_line/results/layers/README.md) · [PTQ 掉点定位](reproduction/formal_line/results/ptq-diagnosis/README.md) · [激活位宽线](reproduction/formal_line/results/activation-sweep/README.md) · [快速复现](#5-快速复现与多机分工)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reproduction/figures/formal-overview-dark.svg">
  <img alt="TUAR、TUAB、TUSL v2 三任务 INT8 测试 AUROC（3 个随机种子均值与范围）：FP32 为 0.87、0.84、0.83，PTQ 为 0.86、0.76、0.82，QAT 为 0.86、0.84、0.83，QAP 为 0.86、0.85、0.86。" src="reproduction/figures/formal-overview-light.svg" width="780">
</picture>

</div>

本页正式线采用受试者独立划分（预训练语料为 TUAB 官方训练集并剔除全部下游测试受试者），固定种子 42 / 43 / 44，使用 Brevitas 伪量化。`QAT` 是本仓库新增的对照（FP32 预训练 + 量化感知微调），用来单独隔离“预训练阶段是否需要带量化”。指标含 ROC 曲线下面积（AUROC）、平衡准确率（BAcc）、期望校准误差（ECE，15 等宽分箱）与比特操作数（BOPs）。

## 立项问题速答

| 立项问题 | 结论与核心数字 | 索引 |
| --- | --- | --- |
| **QAP 从哪个位宽开始优于 PTQ？** | INT8、INT4 两档均已显著优于 PTQ（TUAB 上 AUROC 高 `+0.10`、BAcc 高 `+0.12`；TUSL 上 BAcc 高 `+0.19`；TUAR 重选阈值后持平）。 | [INT8](reproduction/formal_line/results/int8/README.md) · [INT4](reproduction/formal_line/results/int4/README.md) |
| **差距大到要改部署默认吗？** | 要改，但改成 **QAT（量化感知微调）** 就够：QAT 拿到同等恢复，QAP 对 QAT 无稳定优势。 | [§1](#1-正式线主结果三任务-int8w8a8对比) · [§2](#2-正式线主结果三任务-int4w4a8对比) |
| **QAP 值得预训练成本的位宽** | 8 位、4 位都不值得。QAT 共用一份 FP32 预训练（`2.59 h`），QAP 每档位宽需重训一次（折算约 `3.0 h`）。 | [预训练成本](reproduction/formal_line/results/metrics/README.md#预训练成本) |
| **逐层：QAP 在哪里帮助最大？** | **第 2 个 Mamba 块**（占掉点缩减量的 `76%` INT8 / `72%` INT4），尤以反向扫描输出最大；QAT 与 QAP 在该块的激活截断率均 `≤ 0.02%`（PTQ 为 `4%`）。 | [逐层分析](reproduction/formal_line/results/layers/README.md) |
| **部署建议** | 首选 **INT4 + QAT**（比 INT8 省 `47%` 存储、`25%` 合计 BOPs，AUROC 距 FP32 `≤ 0.03`）。仅能做 PTQ 时，用 SmoothQuant 迁移第 2 块离群通道可在 TUAB 上收回 `82%`（INT8）/ `57%`（INT4）掉点。 | [§4](#4-位宽压缩率推理计算量与训练耗时) · [掉点定位](reproduction/formal_line/results/ptq-diagnosis/README.md) |
| **三种基线** | **BF16 / FP16** 与 FP32 持平；**PTQ** 见 §1–§2；**随机初始化**（无预训练直接量化微调）在 TUAB / TUAR 上与预训练持平，在 TUSL（不计第 0 组）上低 `0.07–0.08`。 | [FP16/BF16](reproduction/formal_line/results/metrics/README.md#fp16-和-bf16) · [随机初始化](reproduction/formal_line/results/metrics/README.md#随机初始化) |
| **四项成本与校准指标** | BAcc、ECE 见 §1–§2；每参数有效位数 `8.40 / 4.48 / 2.52 bit`，单片段合计 BOPs `86.4 / 64.7 / 53.8 G`（INT8 / INT4 / 三值，理论值）。 | [指标与成本](reproduction/formal_line/results/metrics/README.md) |
| **与 FEMBA on the Edge 对照** | 趋势一致（论文 TUAB W8A8/W4A8 PTQ `0.77 / 0.71` → QAT `0.88`；本页 PTQ `0.76 / 0.75` → QAT / QAP `0.84–0.85`）。论文用 TUEG 预训练，本页用 TUAB 预训练 2 轮且微调抽 1/5 受试者。 | [上游差异](reproduction/formal_line/results/int8/README.md#与上游协议的差异) |
| **位宽交叉图** | 含三值的完整交叉图见小组报告；本仓库 INT8、INT4 明细见两张结果页。 | [INT8](reproduction/formal_line/results/int8/README.md) · [INT4](reproduction/formal_line/results/int4/README.md) |

## 1. 正式线主结果：三任务 INT8（W8A8）对比

三个随机种子的测试集均值（两位小数；BAcc 在 TUAR / TUAB 下按默认阈值 `0`，TUSL v2 按最大概率分类；`随机初始化` 为不加载预训练直接做量化微调）：

| 任务 | 指标 | FP32 | PTQ | QAT | QAP | 随机初始化 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **TUAR** | AUROC ↑ | 0.87 | 0.86 | 0.86 | 0.86 | **0.87** |
| | 平衡准确率 BAcc ↑ | 0.75 | 0.63 | **0.76** | **0.76** | **0.76** |
| | 校准误差 ECE ↓ | 0.10 | 0.17 | 0.08 | 0.09 | **0.08** |
| **TUAB** | AUROC ↑ | 0.84 | 0.76 | 0.84 | **0.85** | **0.85** |
| | 平衡准确率 BAcc ↑ | 0.76 | 0.65 | 0.76 | **0.77** | **0.77** |
| | 校准误差 ECE ↓ | 0.03 | 0.06 | **0.02** | **0.02** | 0.03 |
| **TUSL v2** | 宏平均 AUROC ↑ | 0.83 | 0.82 | 0.83 | **0.86** | 0.82 |
| **TUSL v2（不计第 0 组）** | 宏平均 AUROC ↑ | **0.91** | 0.89 | 0.90 | 0.90 | 0.83 |
| **TUSL v2** | 宏平均 BAcc ↑ | 0.68 | 0.51 | 0.67 | **0.70** | 0.68 |
| | 校准误差 ECE ↓ | 0.11 | 0.16 | 0.12 | **0.05** | 0.12 |

验证集重选分类阈值后，PTQ 在 TUAR、TUAB 上的 BAcc 分别恢复到 `0.75`、`0.70`（其余组变化 `≤ 0.01`）。四位小数明细与配对差值见 [INT8 完整结果](reproduction/formal_line/results/int8/README.md) 与 [评测指标与成本](reproduction/formal_line/results/metrics/README.md)。

- **PTQ 阈值偏移**：TUAR 种子 42 在默认阈值 `0` 下全判阳性（BAcc `0.50`，ECE `0.33`），验证集重选阈值后恢复到 `0.73`；TUAB 在两种阈值下 PTQ 均比 QAT、QAP 低 `0.07–0.12`。
- **TUSL v2 单受试者效应**：TUSL 四个测试组的第 0 组（55 个事件中 46 个来自同一受试者）主导了全计时的 `+0.03` 差距；剔除第 0 组后，QAT 与 QAP 的宏平均 AUROC 完全相同（`0.9032` vs `0.9032`），且均比随机初始化（`0.8272`）高 `+0.08`。

## 2. 正式线主结果：三任务 INT4（W4A8）对比

权重降至 4 位、激活保持 8 位，QAT 与 QAP 保持 FP32 水平：

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reproduction/figures/formal-overview-int4-dark.svg">
  <img alt="TUAR、TUAB、TUSL v2 三任务 INT4 测试 AUROC（3 个随机种子均值与范围）：FP32 为 0.87、0.84、0.83，PTQ 为 0.84、0.75、0.81，QAT 为 0.87、0.84、0.81，QAP 为 0.85、0.85、0.84。" src="reproduction/figures/formal-overview-int4-light.svg" width="780">
</picture>
</div>

| 任务 | 指标 | FP32 | PTQ | QAT | QAP | 随机初始化 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **TUAR** | AUROC ↑ | **0.87** | 0.84 | **0.87** | 0.85 | **0.87** |
| | 平衡准确率 BAcc ↑ | 0.75 | 0.57 | **0.76** | 0.75 | **0.76** |
| | 校准误差 ECE ↓ | 0.10 | 0.24 | 0.09 | 0.09 | **0.08** |
| **TUAB** | AUROC ↑ | 0.84 | 0.75 | 0.84 | **0.85** | **0.85** |
| | 平衡准确率 BAcc ↑ | 0.76 | 0.65 | **0.77** | **0.77** | **0.77** |
| | 校准误差 ECE ↓ | 0.03 | 0.09 | **0.02** | **0.02** | 0.03 |
| **TUSL v2** | 宏平均 AUROC ↑ | 0.83 | 0.81 | 0.81 | **0.84** | 0.82 |
| **TUSL v2（不计第 0 组）** | 宏平均 AUROC ↑ | **0.91** | 0.88 | 0.90 | 0.90 | 0.83 |
| **TUSL v2** | 宏平均 BAcc ↑ | **0.68** | 0.49 | 0.65 | **0.68** | **0.68** |
| | 校准误差 ECE ↓ | 0.11 | 0.18 | **0.09** | **0.09** | 0.11 |

验证集重选阈值后，PTQ 在 TUAR、TUAB 上的 BAcc 分别恢复到 `0.74`、`0.69`。明细见 [INT4 完整结果](reproduction/formal_line/results/int4/README.md)。

- **存储再省 47%、量化层算力省 50% 而精度不降**：存储 `9.10 → 4.85 MB`，量化层 BOPs `43.6 → 21.8 G`（合计 `86.4 → 64.7 G`），QAT 与 QAP 在三任务上的 AUROC 变化均 `≤ 0.02`。
- **QAT 与 QAP 依然持平**：`QAP − QAT` 的测试 AUROC 在 TUAR 上三个种子皆负（均值 `−0.015`），TUAB 均值 `+0.008`（种子 44 为负），TUSL v2 不计第 0 组仅差 `+0.003`（种子 44 为负）。到 4 位权重为止，共用 FP32 预训练底座的 QAT 已足够。

## 3. TUAB 上 PTQ 掉点定位与 QAP 逐层归因

TUAB 上 W8A8 PTQ 比 FP32 低 `0.08`（`0.84 → 0.76`）。分别拆解权重/激活量化与逐层敏感度：

| TUAB 量化设置 | 权重位宽 | 激活位宽 | 测试 AUROC | 相对 FP32 |
| --- | --- | --- | ---: | ---: |
| **FP32 基线** | FP32 | FP32 | 0.84 | — |
| **只量化 8 位权重（W8A32）** | 8 bit | FP32 | 0.84 | 0.00 |
| **只量化 4 位权重（W4A32）** | 4 bit | FP32 | 0.83 | −0.01 |
| **只量化三值权重（Ternary）** | 2 bit | FP32 | 0.60 | −0.24 |
| **只量化 8 位激活（W32A8）** | FP32 | 8 bit | 0.76 | −0.08 |
| **标准 W8A8 PTQ（95 百分位）** | 8 bit | 8 bit | 0.76 | −0.08 |

两份逐层报告分工不同：**[QAP 逐层分析](reproduction/formal_line/results/layers/README.md)** 对比 PTQ / QAT / QAP 的逐块掉点与截断率（解释为什么量化微调不掉点）；**[PTQ 掉点定位](reproduction/formal_line/results/ptq-diagnosis/README.md)** 针对单模型 PTQ 做 35 处消融与不重训修复（SmoothQuant 迁移、Hadamard 旋转、FP16 回退、MSE/KL 校准）。

- **8 位和 4 位掉在激活，三值掉在权重**：W8A32 不掉点（`0.84`），W4A32 仅掉 `0.01`（`0.83`）；三值只量化权重即跌至 `0.60`，必须靠量化训练恢复。
- **瓶颈在第 2 个 Mamba 块扫描输出，而非长尾最重的第 1 块**：长尾最重的 7 处都在第 1 块，但单独量化掉点 `≤ 0.005`；真正敏感的是**第 2 个 Mamba 块的双向状态空间扫描（SSM）输出**（占 QAP 掉点缩减量的 **76%** INT8 / **72%** INT4）。将该处离群通道平移进下一层 `out_proj` 权重后，不重训的 W8A8 PTQ 恢复至 `0.82`（H128 旋转恢复至 `0.82`）。

## 4. 位宽压缩率、推理计算量与训练耗时

FEMBA-Tiny 共 867 万参数：98.0% 为量化卷积与线性层权重，0.4%（3.1 万）为走 8 位激活量化的位置编码，1.7%（depthwise 卷积、归一化、SSM 参数与偏置）保持 FP32。单次 5 秒片段（`22 × 1280`）的 BOPs 与单卡 RTX 5060 Ti（16 GB）预训练耗时：

| 位宽设置 | 权重×激活 | 有效位数 | 模型存储 | 压缩比 | 量化层 BOPs | 合计 BOPs | 预训练耗时 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| **FP32** | 32 × 32 | 32.00 bit | 34.68 MB | 1.0× | 696.9 G | 739.8 G | 2.59 h（独占） |
| **BF16 / FP16** | 16 × 16 | 16.00 bit | 17.34 MB | 2.0× | 174.2 G | 184.9 G | — |
| **INT8** | 8 × 8 | 8.40 bit | 9.10 MB | **3.8×** | 43.6 G | 86.4 G | 约 3.0 h（折算） |
| **INT4** | 4 × 8 | 4.48 bit | 4.85 MB | **7.1×** | 21.8 G | 64.7 G | 6.24 h（并发） |
| **三值** | 2 × 8 | 2.52 bit | 2.73 MB | **12.7×** | 10.9 G | 53.8 G | — |

`合计 BOPs` 含保持 FP32 的 SSM 扫描与 depthwise 卷积（共 `42.9 G`）。16 位基线中，BF16 零溢出且与 FP32 相差 `≤ 0.0002`；FP16 在第 1 个 Mamba 块反向扫描输出有 `0.03%–0.11%` 的 TUAB 测试片段溢出。下游微调耗时（TUAB `1.93–2.25 h`、TUAR `0.87–1.23 h`、TUSL 每组 `2–3 min`）见 [评测指标与成本](reproduction/formal_line/results/metrics/README.md)。

## 5. 快速复现与多机分工

完整协议、环境变量与多机分跑步骤见 **[reproduction/formal_line/README.md](reproduction/formal_line/README.md)**。

```bash
# 0. 路径：SRC 是打好补丁的上游源码，STUDY 是运行目录，ART 是结果根目录（共享的 FP32 父模型也放这里）
export SRC=<源码目录> STUDY=<运行目录> ART=<结果根目录>

# 1. 安装正式线依赖（Python 3.11，PyTorch 2.7.1+cu128，Brevitas 0.12.1）
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r reproduction/formal_line/requirements-formal.txt

# 2. 检出上游模型源码（a2a4d44）并依次打入三个量化与三值补丁
git clone https://github.com/berumottovv00/BioFoundation.git "$SRC"
git -C "$SRC" checkout a2a4d441485efd8423c44b35d7448a30ccd44437
git -C "$SRC" apply "$PWD/reproduction/teammate_ptq/source.patch" \
                    "$PWD/reproduction/teammate_ptq/quantization-config.patch" \
                    "$PWD/reproduction/teammate_ptq/ternary-model.patch"

# 3. 把脚本拷进运行目录，跑单种子微调、PTQ 七档校准与评估队列（以 INT4 为例，复用共享 FP32 父模型）
#    尚未跑完同种子 QAP 预训练时加 RUN_QAP=no，只跑 QAT（产物目录与 JSON 键沿用内部名 c_* / c）和 PTQ
cp reproduction/teammate_ptq/*.py reproduction/tusl_events/evaluate_events.py \
   reproduction/formal_line/*.sh reproduction/formal_line/*.py "$STUDY"/
cd "$STUDY"
FP32_PARENTS=given SEED=42 BITS=4 SCHEME=uniform \
  TASKS="tuar tuab tusl0-v2 tusl1-v2 tusl2-v2 tusl3-v2" LOG_SUFFIX=-all ./finetune_queue.sh
```

本仓库新增代码与结果均在 `reproduction/` 下（根目录 `models/`、`tasks/`、`config/` 为上游 pulp-bio/BioFoundation 原始代码）：

| 目录 | 内容 |
| --- | --- |
| `reproduction/formal_line/` | 正式线运行脚本、手册与全部结果页（现行） |
| `reproduction/teammate_ptq/` | 共用训练、PTQ、评估脚本与三个源码补丁（其 README 为早期按录音划分的 TUAR 试点） |
| `reproduction/subject_split/`、`reproduction/tusl_events/` | 按受试者划分的 TUAR / TUAB 数据与 TUSL 事件数据生成脚本 |
| `reproduction/RESULTS.md`、`reproduction/PROTOCOL.md` | 2025 公开权重早期复现（已归档） |

## 6. 报告问题与数据速查索引

| 模块 / 指标 | 结果页 | 原始 JSON 与复算脚本 |
| --- | --- | --- |
| **INT8 主结果**（AUROC、AP、配对差值、重抽样） | [INT8 完整结果](reproduction/formal_line/results/int8/README.md) | [summary.json](reproduction/formal_line/results/int8/summary.json) · [provenance.json](reproduction/formal_line/results/int8/provenance.json) · [verify_binary.py](reproduction/formal_line/results/int8/verify_binary.py) |
| **INT4 主结果**（AUROC、AP、配对差值、重抽样、环境） | [INT4 完整结果](reproduction/formal_line/results/int4/README.md) | [summary.json](reproduction/formal_line/results/int4/summary.json) · [provenance.json](reproduction/formal_line/results/int4/provenance.json) · [tusl-bootstrap.json](reproduction/formal_line/results/int4/tusl-bootstrap.json) |
| **平衡准确率（BAcc）**（默认阈值 `0` vs 验证集阈值） | [评测指标与成本 §平衡准确率](reproduction/formal_line/results/metrics/README.md#平衡准确率) | [int8/summary.json](reproduction/formal_line/results/int8/summary.json) · [int4/summary.json](reproduction/formal_line/results/int4/summary.json) |
| **期望校准误差（ECE）**（15 等宽分箱） | [评测指标与成本 §校准误差](reproduction/formal_line/results/metrics/README.md#校准误差) | [calibration-error.json](reproduction/formal_line/results/metrics/calibration-error.json) · [calibration_error.py](reproduction/formal_line/calibration_error.py) |
| **16 位基线**（BF16 / FP16 重放与溢出统计） | [评测指标与成本 §FP16 和 BF16](reproduction/formal_line/results/metrics/README.md#fp16-和-bf16) | [half-precision.json](reproduction/formal_line/results/metrics/half-precision.json) · [half_precision.py](reproduction/teammate_ptq/half_precision.py) |
| **随机初始化基线**（INT8 / INT4 无预训练直接微调） | [评测指标与成本 §随机初始化](reproduction/formal_line/results/metrics/README.md#随机初始化) | [random-init.json](reproduction/formal_line/results/metrics/random-init.json) · [run_random_init.sh](reproduction/formal_line/run_random_init.sh) |
| **模型成本**（有效位数、存储、BOPs） | [评测指标与成本 §有效位数](reproduction/formal_line/results/metrics/README.md#每参数有效位数和存储) | [model-cost.json](reproduction/formal_line/results/metrics/model-cost.json) · [model_cost.py](reproduction/formal_line/model_cost.py) |
| **训练耗时与卡型**（预训练与各任务微调） | [评测指标与成本 §预训练成本](reproduction/formal_line/results/metrics/README.md#预训练成本) | [int4/provenance.json](reproduction/formal_line/results/int4/provenance.json) |
| **逐样本测试分数**（文件布局与字段格式） | [INT8 完整结果 §文件](reproduction/formal_line/results/int8/README.md#文件) | 组内按压缩包分发，格式见该节 |
| **QAP 逐层分析**（按块 + 第 2 块逐处 + SQNR/截断率） | [QAP 逐层分析](reproduction/formal_line/results/layers/README.md) | [layers/*.json](reproduction/formal_line/results/layers/) · [layer_comparison.py](reproduction/teammate_ptq/layer_comparison.py) |
| **PTQ 掉点诊断与修复**（迁移、旋转、FP16 回退、MSE/KL） | [PTQ 掉点定位](reproduction/formal_line/results/ptq-diagnosis/README.md) | [ptq-diagnosis/summarize.py](reproduction/formal_line/results/ptq-diagnosis/summarize.py) · [channel_migration.py](reproduction/teammate_ptq/channel_migration.py) · [rotation_ptq.py](reproduction/teammate_ptq/rotation_ptq.py) |
| **激活位宽线**（W8A6 / W8A5 / W8A4 与 95 百分位起步） | [激活位宽线](reproduction/formal_line/results/activation-sweep/README.md) | [summary.json](reproduction/formal_line/results/activation-sweep/summary.json) · [provenance.json](reproduction/formal_line/results/activation-sweep/provenance.json) · [run_c_start.sh](reproduction/formal_line/run_c_start.sh) |
| **正式线协议与队列**（[subject_split](reproduction/subject_split/)、[tusl_events](reproduction/tusl_events/)、[teammate_ptq](reproduction/teammate_ptq/README.md)） | [正式线复现手册](reproduction/formal_line/README.md) · [2025 权重归档](reproduction/RESULTS.md) | [finetune_queue.sh](reproduction/formal_line/finetune_queue.sh) · [summarize_formal.py](reproduction/formal_line/summarize_formal.py) |

## 上游来源与致谢

- **量化模型与 QAP 基线**：基于 [berumottovv00/BioFoundation](https://github.com/berumottovv00/BioFoundation/tree/a2a4d441485efd8423c44b35d7448a30ccd44437)（`a2a4d44`）。
- **FEMBA 原始论文与公开实现**：[pulp-bio/BioFoundation](https://github.com/pulp-bio/BioFoundation/tree/d88596590f3bd3fce573be07646b7d3977ce7bcc) · [FEMBA（EMBC 2025）](https://arxiv.org/abs/2502.06438v2) · [FEMBA on the Edge（IEEE TBME 2026）](https://arxiv.org/abs/2603.26716v1) · [Apache-2.0 License](LICENSE)。
