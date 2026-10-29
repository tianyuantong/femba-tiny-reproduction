# FEMBA-Tiny 复现与量化实验

比较训练后量化（PTQ）、FP32 预训练后量化微调（C），以及量化预训练后量化微调（QAP）。正式线方案覆盖 TUAR、TUAB、TUSL 三个任务，采用受试者划分，设置 int8（W8A8）、int4（W4A8）和三值权重（−1 / 0 / +1，激活 8 位），每档与共用的 FP32 基线对照，种子为 42 / 43 / 44。W/A 表示权重／激活位宽，量化实验使用 Brevitas 伪量化。

## 仓库已有内容

| 内容 | 说明与入口 |
| --- | --- |
| 正式线 FP32 / INT8 结果 | 受试者划分，TUAR、TUAB、TUSL 三个任务，FP32、PTQ、C、QAP 四组各三个种子，见 [INT8 结果](reproduction/formal_line/results/int8/README.md) |
| TUAB 上 PTQ 的掉点定位 | INT8 PTQ 的损失来自哪些激活、改用 FP16 能收回多少，另有 MSE / KL 校准和 INT4、三值权重的对照，见 [PTQ 掉点定位](reproduction/formal_line/results/ptq-diagnosis/README.md) |
| TUAR INT8 试点结果 | 按录音划分的三个种子，包含 FP32、PTQ、C、QAP 四组及配对分析，见 [INT8 团队小结](reproduction/teammate_ptq/README.md) |
| 正式线复现流程 | 三个任务的数据构建、环境、预训练、微调、PTQ、评估、汇总与多机分工，见 [正式线运行说明](reproduction/formal_line/README.md) |

## 从哪里开始

从 **`baseline-2025` 分支的 [reproduction/formal_line/README.md](reproduction/formal_line/README.md)** 开始。

1. **准备环境和数据**：按[环境与数据](reproduction/formal_line/README.md#环境与数据)安装依赖、生成数据并核对清单。仓库提供数据构建与校验脚本，数据文件和检查点另行交付，不随 Git 仓库分发。
2. **准备模型源码**：按[独立准备源码](reproduction/teammate_ptq/QUANTIZATION.md#独立准备源码)检出上游 `a2a4d44`，依次应用 `source.patch`、`quantization-config.patch`、`ternary-model.patch`。正式线训练参数以正式线运行说明为准。
3. **运行自己负责的位宽**：按[多台机器分跑](reproduction/formal_line/README.md#多台机器分跑)设置路径、接收共用 FP32 父检查点，再运行 INT4 或三值的预训练、微调和评估；三值先做冒烟测试。结果用 [summarize_formal.py](reproduction/formal_line/summarize_formal.py) 汇总。

TUSL 使用标注事件三分类和四折评估；v2 验证集规则是看过 v1 测试结果后的探索性修订，协议要求保留 v1 对照。具体划分和指标见[正式线协议](reproduction/formal_line/README.md#协议)。

## 已完成：正式线 FP32 与 INT8

受试者划分，种子 42 / 43 / 44。下表是测试 AUROC 的均值 ± 样本标准差；TUSL 是协议 v2 下事件级宏平均 AUROC 的四折平均。

| 任务 | FP32 | PTQ INT8 | C INT8 | QAP INT8 |
| --- | ---: | ---: | ---: | ---: |
| TUAR | 0.8663 ± 0.0064 | 0.8552 ± 0.0177 | 0.8642 ± 0.0051 | 0.8608 ± 0.0133 |
| TUAB | 0.8391 ± 0.0083 | 0.7560 ± 0.0473 | 0.8392 ± 0.0126 | 0.8510 ± 0.0069 |
| TUSL v2 | 0.8339 ± 0.0292 | 0.8194 ± 0.0228 | 0.8263 ± 0.0324 | 0.8609 ± 0.0280 |

PTQ 在三个任务上均值都最低，TUAB 上比 FP32 低 0.083。QAP 对 PTQ 在 TUAB、TUSL v2 上三个种子都领先，TUAR 上基本持平。三个任务都没有 QAP 稳定优于 C 的证据：TUSL v2 上的 +0.035 全部来自一名受试者主导的一折。

[完整结果与结论](reproduction/formal_line/results/int8/README.md) · [逐种子结果](reproduction/formal_line/results/int8/summary.json) · [来源记录](reproduction/formal_line/results/int8/provenance.json)

## 已完成：TUAR INT8 试点

**本节只有一个 TUAR 数据集，按录音划分，未按受试者划分。** 测试集为 7,211 个窗口；下表是 seeds 42 / 43 / 44 的均值 ± 样本标准差。

| 方法 | 测试 AUROC |
| --- | ---: |
| FP32 | 0.9045 ± 0.0095 |
| PTQ INT8 | 0.8707 ± 0.0082 |
| C：FP32 预训练＋量化微调 | 0.8940 ± 0.0168 |
| QAP B：量化预训练＋量化微调 | 0.9021 ± 0.0055 |

QAP B 在三个种子上均优于 PTQ。与 C 的严格同初始化配对只有 seed43/44，QAP B−C 的 AUROC 差为 **−0.0013 ± 0.0036**，未观察到量化预训练的一致收益；QAP42 使用队友检查点的完整复测值。

[完整指标与结论](reproduction/teammate_ptq/README.md) · [逐种子结果与来源](reproduction/teammate_ptq/c-seed-results.json) · [试点复现命令](reproduction/teammate_ptq/TRAINING.md)

## 历史实验：2025 公开模型复现

早期实验使用 FEMBA 2025 公开代码和预训练权重，在 TUAR 的 7,090 个测试窗口上进行单种子 seed42 对照，包括 FP32 标签／学习率复现、FP16、PTQ 和 H128 Hadamard 旋转。该阶段使用 FP32 QDQ 模拟，其量化位置和评估口径与上述 INT8 试点不同，结果分别保留。

在该阶段的配置下，W8A32 精度接近 FP32，57 处激活量化的 W8A8 / W4A8 得到 AUROC 0.5；FP16 三种路径均在首批验证出现 NaN。完整数值、曲线与诊断见[历史实验结果](reproduction/RESULTS.md)，运行步骤见[历史复现协议](reproduction/PROTOCOL.md)。

## 上游来源

- **INT8 试点与正式线**：模型及 QAP 流水线来自 [berumottovv00/BioFoundation](https://github.com/berumottovv00/BioFoundation/tree/a2a4d441485efd8423c44b35d7448a30ccd44437)，本仓库提供复现脚本、量化配置补丁、数据划分与评估流程。
- **早期公开模型复现**：基于 [pulp-bio/BioFoundation 的 2025 快照](https://github.com/pulp-bio/BioFoundation/tree/d88596590f3bd3fce573be07646b7d3977ce7bcc)和 [PulpBio/FEMBA 预训练权重](https://huggingface.co/PulpBio/FEMBA)。
- **论文与许可证**：[FEMBA（2025）](https://arxiv.org/abs/2502.06438v2) · [Apache-2.0](LICENSE)。
