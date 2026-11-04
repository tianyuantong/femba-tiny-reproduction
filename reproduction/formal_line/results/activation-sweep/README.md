# 正式线：激活位宽线（W8A6、W8A5、W8A4）

TUAB，受试者划分，种子 42 / 43 / 44。数据、FP32 父模型、微调和 PTQ 流程都和 [INT8 结果](../int8/README.md)相同，只把激活从 8 位降下来。W8A6 表示权重 8 位、激活 6 位。

正式线的三档（int8、int4、三值）激活都是 8 位。[FEMBA on the Edge（2026）](https://arxiv.org/abs/2603.26716v1)里 QAT 在 W2A8 仍有 AUROC 0.88，到 W4A4 掉到 0.69，所以这一页单独降激活位宽，找 C 和 QAP 的临界点。

## 判据

下面的规则在开跑前定下，看到测试结果后不改。

| 项目 | 规定 |
|---|---|
| 主指标 | TUAB 测试 AUROC，同一种子内和 FP32 比 |
| 过了临界点 | 某一档 C 的三种子均值比 FP32 低超过 0.01，且三个种子都低于 FP32。从 A6 往下第一个满足的档就是临界点 |
| 第一步 | W8A6、W8A5、W8A4 各跑 C 和 PTQ，共用正式线的 FP32 预训练和微调 |
| 第二步 | 在临界点和它上面一档跑 QAP，预训练和微调位宽相同 |
| QAP 的稳定收益 | QAP − C 的三种子均值至少 0.05，且三个种子都为正。W8A8 上 QAP − C 的三种子 95% 范围约 ±0.048，这个幅度才测得出 |
| W8A4 也没过临界点 | 往下怎么走另定，不算本页协议 |

## 运行

`ART` 是新的输出目录，每个种子的 `fp32` 链到正式线目录，这样 C 和 PTQ 共用同一份 FP32，正式线目录不写入。其余环境变量见[正式线运行说明](../../README.md#环境与数据)。

```bash
FORMAL=/root/work/femba-artifacts/formal-20261025
mkdir -p "$ART/logs"
for seed in 42 43 44; do mkdir -p "$ART/seed$seed"; ln -s "$FORMAL/seed$seed/fp32" "$ART/seed$seed/fp32"; done
# 第一步：先跑 A4，它离 A8 最远，最早给出结论
for abits in 4 6 5; do
  for seed in 42 43 44; do
    FP32_PARENTS=given RUN_QAP=no SEED=$seed BITS=8 ABITS=$abits TASKS=tuab LOG_SUFFIX=-activation ./finetune_queue.sh || exit 1
  done
done
```
