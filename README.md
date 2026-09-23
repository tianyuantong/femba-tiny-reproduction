# FEMBA-Tiny 2025 复现与精度实验

基于 FEMBA 2025 年公开代码和预训练权重，复现 TUAR 二分类基线，并比较 FP16、PTQ 和 Hadamard 旋转后的结果。

单种子 seed 42 · 7,090 个测试窗口 · RTX 5060 Ti · 量化采用 FP32 QDQ 模拟。

## FP32 复现

![FP32 复现结果与论文 Table IV 的 AUROC 对照](reproduction/figures/fp32-gap.svg)

| 配置 | 标签／AdamW 学习率 | AUROC | AP |
| --- | --- | ---: | ---: |
| 公开代码配方 | 18 类、占比 ≥30%／5e-4 | 0.923889 | 0.933093 |
| 论文标签对照 | 13 类、任一出现／5e-4 | 0.913040 | 0.944898 |
| 学习率对照 | 13 类、任一出现／1e-4 | 0.898821 | 0.938437 |
| 2025 论文 Table IV | TUAR 二分类 | 0.937 ± 0.008 | 0.912 ± 0.010 |

论文标签组保留代码的归一化、AdamW 和模型实现。

## 精度对照

![各量化配置的测试 AUROC](reproduction/figures/precision-ladder.svg)

统一使用按最低验证 loss 选定的 1e-4 检查点，FP32 AUROC 为 **0.8988**。
W/A 表示权重／激活位宽；A8 使用 57 处量化位置、浮点尺度。

| 配置 | 未旋转 AUROC | H128 旋转 AUROC |
| --- | ---: | ---: |
| W8A32 | 0.8999 | 0.9001 |
| W4A32 | 0.8449 | 0.8219 |
| W8A8 | 0.5000 | 0.5000 |
| W4A8 | 0.5000 | 0.5000 |

FP16 的直接转换、状态保留和 AMP 三种路径均在首批验证出现 NaN。

W8A32 精度接近 FP32，已测 A8 配置均预测为正类。
Linear22/W8A8 的首个前向分支 `out_proj` 输入在静态 max 尺度下，99.9964% 的元素从非零量化为零。
H128 将五处输出投影输入的校准峰值缩小 4.3–11.1 倍，Rot-A8 分类仍未恢复。

[完整结果](reproduction/RESULTS.md) · [复现步骤](reproduction/PROTOCOL.md)

## 来源

基于 [pulp-bio/BioFoundation 的 2025 快照](https://github.com/pulp-bio/BioFoundation/tree/d88596590f3bd3fce573be07646b7d3977ce7bcc)
和 [PulpBio/FEMBA 预训练权重](https://huggingface.co/PulpBio/FEMBA)，论文见
[FEMBA（2025）](https://arxiv.org/abs/2502.06438v2)。上游代码采用 [Apache-2.0](LICENSE)。
仓库提供代码和结果摘要；数据集、权重及逐样本预测保留在本地。
