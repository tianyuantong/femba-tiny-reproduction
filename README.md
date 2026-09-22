# FEMBA-Tiny 2025 复现与精度实验

基于 FEMBA 2025 年公开代码和预训练权重，在 TUAR 上复现二分类基线，并比较低精度推理的质量变化。

**W8A32 保持基线精度；W4A32 与 A8 明显退化。**

单种子 seed 42 · 7,090 个测试窗口 · RTX 5060 Ti · 量化采用 FP32 QDQ 模拟。

| 版本 | AUROC | AP | 结果 |
| --- | ---: | ---: | --- |
| FP32 | 0.898821 | 0.938437 | 当前母模型 |
| W8A32 | 0.899945 | 0.938976 | 在预设精度容差内 |
| W4A32 | 0.844892 | 0.903830 | 明显退化 |
| W8A8，浮点尺度 | 0.500000 | 0.577433 | 全部预测为正类 |
| W4A8，浮点尺度 | 0.500000 | 0.577433 | 全部预测为正类 |
| W8A8，二次幂尺度 | 0.500000 | 0.577433 | 全部预测为正类 |
| FP16：直接转换／状态保留／AMP | — | — | 首批验证出现 NaN |
| Rot-W8A32／W4A32／W8A8／W4A8 | — | — | H128 逐层一致性检查未通过 |
| 论文 Table IV | 0.937 ± 0.008 | 0.912 ± 0.010 | 参考值 |

母模型按最低验证损失选择：AdamW 学习率从 5e-4 降至 1e-4 后，验证损失从
0.426979 降至 0.393690，测试 AUROC 从 0.913040 降至 0.898821。

[完整结果与关键对照](reproduction/RESULTS.md) · [复现步骤与实验口径](reproduction/PROTOCOL.md) ·
[精度结果 JSON](reproduction/precision_results.json)

## 来源

基于 [pulp-bio/BioFoundation 的 2025 快照](https://github.com/pulp-bio/BioFoundation/tree/d88596590f3bd3fce573be07646b7d3977ce7bcc)
和 [PulpBio/FEMBA 预训练权重](https://huggingface.co/PulpBio/FEMBA)，论文见
[FEMBA（2025）](https://arxiv.org/abs/2502.06438v2)。上游代码采用 [Apache-2.0](LICENSE)。
仓库提供代码和结果摘要；数据集、权重及逐样本预测保留在本地。
