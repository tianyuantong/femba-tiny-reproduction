# 队友模型的 FP32、PTQ 与 QAP 对照

使用 [队友代码的固定提交](https://github.com/berumottovv00/BioFoundation/tree/a2a4d441485efd8423c44b35d7448a30ccd44437) 和同一份 TUAR 数据。
模型为 ARES 版本，8,670,626 个参数，实际 `dt_rank=96`。seed 42，预训练 10 轮、微调 5 轮，按最低验证 loss 保存检查点。

## 结果

完整测试集 7,211 个窗口，负类 3,304、正类 3,907。四组使用同一套预处理、样本顺序与指标。
两份 PTQ 来自同一 FP32 检查点；QAP B 使用队友预训练及量化微调的检查点。

| 模型 | AUROC | 默认阈值 BAcc | 验证集阈值 BAcc |
| --- | ---: | ---: | ---: |
| FP32 | 0.9140 | 83.27% | 83.43% |
| 原 PTQ INT8 | 0.6925 | 49.28% | 66.73% |
| P99 PTQ INT8 | 0.8717 | 50.09% | 76.86% |
| 队友 QAP B | 0.9017 | 81.98% | 82.18% |

AUROC/AP 统一使用两类 logit 的差值 `z1-z0`；上游使用单列 `z1`。QAP B 行来自收到的检查点在本机 WSL 的完整测试。原生指标和队友报告参考值一并保存在结果 JSON 中。

P99 恢复了大部分排序能力，但输出偏向正类：真实负类的正类概率中位数从 FP32 的 0.077 升到 0.830，默认阈值 0.5 下判正类比例为 99.82%。只在验证集选择阈值后，P99 的概率阈值为 0.886678，测试判正类比例降到 62.77%，Accuracy 为 77.74%。量化权重和尺度均未改变。

四组统一在 6,426 个验证窗口上最大化 Balanced Accuracy；并列时取候选中离默认阈值最近、再取较低的阈值。阈值脚本先保存四组选择，再读取测试 logits 计算对照指标。调整阈值不改变 AUROC/AP。

[完整数值与来源](results.json) 包含 AP、Accuracy、混淆矩阵、阈值和数据/权重哈希。

## 设置

| 项目 | 设置 |
| --- | --- |
| 训练 | batch 32；预训练 AdamW 1e-4；微调 AdamW 5e-5、weight decay 0.5、warmup 1 轮、分层衰减 0.75 |
| 输入 | 22×1,280；固定归一化 `(x+20)/(40+1e-8)`；HDF5 样本缓存关闭 |
| PTQ | 同一份 FP32 权重；seed 42 固定抽取 512 个训练窗口；35 个激活、18 个权重量化器全部开启 |
| 校准 | 原生 Brevitas 百分位统计，按调用求平均；原 PTQ 为 99.999，P99 为 99.0 |
| 推理 | RTX 5060 Ti；batch 32；FP32 执行；TF32 开启；INT8 使用原生 Brevitas 伪量化 |

P99 沿用此前在 99.999、99.99、99.9、99.0 四组中按验证集原生 AUROC 选出的方案。本轮固定它的权重和尺度，只比较指标口径与分类阈值。

## 运行

在实验环境中固定队友代码并应用补丁。补丁只增加量化开关和原生 PTQ 校准函数，保留模型结构。

```bash
git clone https://github.com/berumottovv00/BioFoundation.git teammate-source
git -C teammate-source checkout a2a4d441485efd8423c44b35d7448a30ccd44437
git -C teammate-source apply /path/to/this-repo/reproduction/teammate_ptq/source.patch
```

补丁保留 FP32 检查点中“尚未统计”的激活状态；加载到量化模型后会重新统计，供 C 组量化微调使用。上表对应修复前的补丁，版本记录见 `results.json` 的 `source.model_patch_revision`。

下列路径由运行者提供；数据目录包含 `train.h5`、`val.h5`、`test.h5`。输出目录使用仓库外的新目录。

```bash
study=/path/to/this-repo/reproduction/teammate_ptq
teammate_source=/path/to/teammate-source
data=/path/to/TUAR_data
artifacts=/path/to/artifacts

python "$study/calibrate.py" --source "$teammate_source" --data "$data" \
  --checkpoint "$artifacts/fp32.ckpt" --percentile 99.999 --output "$artifacts/ptq-original"
python "$study/calibrate.py" --source "$teammate_source" --data "$data" \
  --checkpoint "$artifacts/fp32.ckpt" --percentile 99.0 --output "$artifacts/ptq-p99"
python "$study/evaluate.py" --source "$teammate_source" --data "$data" \
  --fp32 "$artifacts/fp32.ckpt" --ptq-original "$artifacts/ptq-original/ptq_int8.ckpt" \
  --ptq-p99 "$artifacts/ptq-p99/ptq_int8.ckpt" --qap-b "$artifacts/qap-b.ckpt" \
  --output "$artifacts/captures"
python "$study/threshold_control.py" --captures "$artifacts/captures" \
  --output "$artifacts/thresholds"
```

`calibrate.py` 只读取训练数据，保存后严格重载，核对浮点参数、53 个量化尺度和探针输出。`evaluate.py` 保存有序 logits；`threshold_control.py` 只用验证输出选择阈值。

训练配置及完整覆盖参数保存在 `results.json` 的 `training` 字段。环境版本见 `runtime` 字段。数据集、检查点、逐样本输出和原始日志保存在本地。

```bash
FEMBA_SOURCE="$teammate_source" python -m unittest discover -s "$study" -p 'test_*.py' -v
```
