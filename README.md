<div align="center">

# FEMBA-Tiny 脑电基础模型量化与位宽临界点研究

在可穿戴脑电基础模型 FEMBA-Tiny（双向 Mamba 架构，867 万参数）上对比训练后量化（PTQ）、FP32 预训练 + 量化微调（C 组）与量化感知预训练（QAP）。下游任务是 TUAR、TUAB、TUSL，位宽分 INT8、INT4、三值三档，本页结果是 INT8。

**TUAB 上 QAP 0.85、PTQ 0.76 · TUSL v2 上 QAP 0.86、PTQ 0.82 · INT8 存储 9.19 MB（3.8×）**

![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![PyTorch 2.7](https://img.shields.io/badge/pytorch-2.7.1%2Bcu128-ee4c2c)
![Brevitas 0.12.1](https://img.shields.io/badge/brevitas-0.12.1-008080)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-lightgrey.svg)](LICENSE)

[快速复现与分工](#快速复现与多机分工) · [INT8 完整结果](reproduction/formal_line/results/int8/README.md) · [评测指标与成本](reproduction/formal_line/results/metrics/README.md) · [PTQ 掉点定位](reproduction/formal_line/results/ptq-diagnosis/README.md)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reproduction/figures/formal-overview-dark.svg">
  <img alt="TUAR、TUAB、TUSL v2 三任务 INT8 测试 AUROC（3 个随机种子均值与范围）：FP32 为 0.87、0.84、0.83，PTQ 为 0.86、0.76、0.82，C 为 0.86、0.84、0.83，QAP 为 0.86、0.85、0.86。" src="reproduction/figures/formal-overview-light.svg" width="780">
</picture>

</div>

本页的正式线实验采用受试者独立划分（预训练语料为 TUAB 官方训练集并剔除全部下游测试受试者），固定随机种子 42 / 43 / 44，在单张 RTX 5060 Ti（16 GB）上使用 Brevitas 伪量化完成。

## 1. 正式线主结果：三任务 INT8（W8A8）对比

下表汇总三个随机种子的测试集均值（两位小数）：包含 ROC 曲线下面积（AUROC）、平衡准确率（BAcc，斜杠前是默认阈值，斜杠后是验证集选定的阈值，TUSL 按最大概率分类）与期望校准误差（ECE，15 等宽分箱，越低越好）。

| 任务 | 指标 | FP32 基线 | 训练后量化 (PTQ) | FP32 预训练 + 量化微调 (C) | 量化预训练 + 量化微调 (QAP) |
| --- | --- | ---: | ---: | ---: | ---: |
| **TUAR**（伪迹检测） | AUROC | 0.87 | 0.86 | 0.86 | 0.86 |
| | BAcc（默认 / 验证阈值） | 0.75 / 0.75 | 0.63 / 0.75 | 0.76 / 0.75 | 0.76 / 0.75 |
| | 校准误差 ECE ↓ | 0.10 | 0.17 | **0.08** | 0.09 |
| **TUAB**（异常检测） | AUROC | 0.84 | 0.76 | 0.84 | **0.85** |
| | BAcc（默认 / 验证阈值） | 0.76 / 0.76 | 0.65 / 0.70 | 0.76 / 0.76 | **0.77 / 0.77** |
| | 校准误差 ECE ↓ | 0.03 | 0.06 | **0.02** | **0.02** |
| **TUSL v2**（慢波三分类） | 宏平均 AUROC | 0.83 | 0.82 | 0.83 | **0.86** |
| | 宏平均 BAcc | 0.68 | 0.51 | 0.67 | **0.70** |
| | 校准误差 ECE ↓ | 0.11 | 0.16 | 0.12 | **0.05** |

四位小数明细、逐种子标准差与配对差值见 [INT8 完整结果](reproduction/formal_line/results/int8/README.md) 与 [评测指标与成本](reproduction/formal_line/results/metrics/README.md)。

- **PTQ 的 AUROC 和校准在三个任务上都最差**：TUAB 上 PTQ 的 AUROC 比 FP32 低 0.08（`0.7560` vs `0.8391`）；TUAR 上 PTQ 虽有 `0.86` AUROC，但种子 42 在默认阈值下把全部片段判为阳性（默认 BAcc 跌至 `0.50`，ECE 达 `0.33`），用验证集重选阈值后恢复到 `0.73`。
- **量化微调（C 与 QAP）即可抹平 INT8 掉点**：在 TUAB 和 TUSL v2 上，QAP 在全部三个种子上均优于 PTQ（AUROC 平均高 `+0.10` 和 `+0.04`）。
- **INT8 下尚无 QAP 稳定优于 C 组的证据**：TUAR 上差 `−0.003`，TUAB 上差 `+0.012`，都跨种子正负不一；TUSL v2 上 QAP 对 C 的 `+0.03` AUROC 优势全部来自第 0 测试组（其中 1 名受试者贡献了 55 个测试事件中的 46 个，剔除该测试组后两组差值为 `0.00`）。

## 2. TUAB 上 PTQ 掉点定位

针对 TUAB 上 W8A8 PTQ 掉点 `0.08`（`0.8391 → 0.7560`），通过解耦权重、激活量化器和逐层敏感度扫描定位来源：

| TUAB 设置（3 种子均值） | 权重位宽 | 激活位宽 | 测试 AUROC | 相对 FP32 变化 |
| --- | --- | --- | ---: | ---: |
| **FP32 基线** | FP32 | FP32 | 0.84 | — |
| **只量化权重（W8 / W4 / 三值）** | 8 / 4 / 2 | FP32 | 0.84 / 0.83 / 0.60 | 0.00 / −0.01 / −0.24 |
| **只量化激活（FP32 权重 + A8）** | FP32 | 8 | 0.76 | −0.08 |
| **标准 W8A8 PTQ（95 百分位）** | 8 | 8 | 0.76 | −0.08 |

逐层排名、长尾比值和不重训的修复实验见 [PTQ 掉点定位](reproduction/formal_line/results/ptq-diagnosis/README.md)。

- **掉点 100% 来自 8 位激活，三值瓶颈则在权重**：W8A8 下只量化权重几乎无损（`0.8390`），W4A8 权重仅多掉 `0.01`（`0.8305`）；而三值只量化权重即跌至 `0.60`，说明三值必须依赖量化训练（C / QAP）。
- **长尾最重不等于对量化敏感**：全模型长尾最极端的 7 处张量（全部位于第 1 个 Mamba 块，99.999 与 90 百分位比值高达 35 万–75 万倍）单独量化时验证集掉点均小于 `0.005`；真正敏感的是**第 2 个 Mamba 块的双向状态空间扫描（SSM）输出**。

## 3. 位宽压缩率、推理计算量与预训练成本

FEMBA-Tiny 共 `8,670,626` 个参数，其中 `8,496,530` 个（98.0%）为量化卷积与线性层权重，其余 `174,096` 个（2.0%，含 4 个 depthwise 卷积、归一化层、状态空间参数与偏置）及 18 个权重量化尺度保持 FP32。单次 5 秒脑电片段（`22 × 1280`）的计算量与 RTX 5060 Ti（16 GB）上的预训练耗时如下：

| 位宽设置 | 权重 × 激活 | 每参数有效位数 | 模型存储体积 | 压缩比 | 单片段比特操作数 (BOPs) | 预训练耗时（2 轮 TUAB） |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| **FP32** | 32 × 32 | 32.00 bit | 34.68 MB | 1.0× | 696.9 G | 2.59 h（独占 GPU） |
| **INT8** | 8 × 8 | 8.48 bit | 9.19 MB | **3.8×** | 43.6 G（16.0×） | 约 3.0 h（独占折算；和微调同跑时 6.60–6.77 h） |
| **INT4** | 4 × 8 | 4.56 bit | 4.94 MB | **7.0×** | 21.8 G（32.0×） | 6.24 h（和微调同跑） |
| **三值 (Ternary)** | 2 × 8 | 2.60 bit | 2.82 MB | **12.3×** | 10.9 G（64.0×） | — |

Brevitas 是浮点模拟，这里比较的是计算量，不是部署延迟。计算公式与脚本见 [评测指标与成本](reproduction/formal_line/results/metrics/README.md)（可由 `reproduction/formal_line/model_cost.py` 与 `calibration_error.py` 直接复算）。

## 4. 快速复现与多机分工

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

## 5. 仓库目录与历史实验归档

| 路径 | 一句话说明 |
| --- | --- |
| [reproduction/formal_line/README.md](reproduction/formal_line/README.md) | 正式线三任务受试者独立划分协议、队列脚本（`finetune_queue.sh`、`run_random_init.sh`）与多机分跑手册 |
| [reproduction/formal_line/results/int8/](reproduction/formal_line/results/int8/README.md) | 正式线 FP32 与 INT8 三种子完整指标、`summary.json`、`provenance.json`、TUSL 重抽样检验与独立复核脚本 |
| [reproduction/formal_line/results/metrics/](reproduction/formal_line/results/metrics/README.md) | 平衡准确率（BAcc）、校准误差（ECE）、每参数有效位数、BOPs 推理计算量与预训练耗时汇总 |
| [reproduction/formal_line/results/ptq-diagnosis/](reproduction/formal_line/results/ptq-diagnosis/README.md) | TUAB 上 PTQ 掉点拆解、35 处激活敏感度排序，以及 FP16 回退和通道迁移两个修复实验 |
| [reproduction/subject_split/](reproduction/subject_split/) | 按受试者切分 TUAR / TUAB 并从预训练语料剔除下游测试受试者的数据构建工具 |
| [reproduction/tusl_events/](reproduction/tusl_events/) | TUSL 事件级三分类构建、四组交叉验证划分与事件级评估脚本 |
| [reproduction/teammate_ptq/](reproduction/teammate_ptq/README.md) | 核心训练与评估驱动（`train_stage.py`、`calibrate.py`、`evaluate_one.py`、三个源码补丁）及早期按录音划分的 TUAR INT8 试点记录 |
| [reproduction/RESULTS.md](reproduction/RESULTS.md) | 早期基于 2025 公开权重与 `mamba-ssm` 的 TUAR 单种子复现记录（含 FP16 首 batch 溢出 NaN 诊断与 Hadamard 旋转对照） |

## 上游来源与致谢

- **量化模型与 QAP 基线**：基于 [berumottovv00/BioFoundation](https://github.com/berumottovv00/BioFoundation/tree/a2a4d441485efd8423c44b35d7448a30ccd44437)（`a2a4d44`）。
- **FEMBA 原始论文与公开实现**：[pulp-bio/BioFoundation](https://github.com/pulp-bio/BioFoundation/tree/d88596590f3bd3fce573be07646b7d3977ce7bcc) · [FEMBA（EMBC 2025）](https://arxiv.org/abs/2502.06438v2) · [FEMBA on the Edge（IEEE TBME 2026）](https://arxiv.org/abs/2603.26716v1) · [Apache-2.0 License](LICENSE)。
