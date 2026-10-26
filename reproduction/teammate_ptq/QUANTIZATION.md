# 共用位宽配置与三值接口

权重、激活分别使用 `weight_bit_width` / `activation_bit_width`，默认均为 8；对应的 `weight_quant_scheme` / `activation_quant_scheme` 默认为 `uniform`；权重另支持 `ternary`（见下），激活只支持 `uniform`。位宽必须为 2–16 的整数，不能是布尔值。以下命令控制 Brevitas 伪量化 proxy；本页不报告新的 INT4 或三值准确率，也不提供低位宽部署内核。

量化覆盖保持原样：分类器 **35A / 18W**，预训练编码器 **34A / 17W**；4 个 depthwise 卷积权重、归一化与预训练 decoder 保持 FP32，实际 `dt_rank=96`。默认 W8A8 保持原参数和计算路径。完整配置存入训练配置、检查点和量化检查记录；旧检查点缺少配置字段时按 uniform W8A8 读取。量化检查点不能改位宽或 scheme 后加载；FP32 父模型允许转换到指定目标配置，C 的 observer 从新统计开始。

## 独立准备源码

保留 [source.patch](source.patch) 的历史版本，在全新副本上依次应用两个 patch。环境沿用 [历史运行版本](seed-results.json) 与 [训练说明](TRAINING.md)，Brevitas 固定为 0.12.1。

```bash
study=/path/to/this-repo/reproduction/teammate_ptq
teammate_source=/path/to/new-quantization-source
data=/path/to/TUAR_data
artifacts=/path/to/new-quantization-artifacts

git clone https://github.com/berumottovv00/BioFoundation.git "$teammate_source"
git -C "$teammate_source" checkout --detach a2a4d441485efd8423c44b35d7448a30ccd44437
git -C "$teammate_source" apply --check "$study/source.patch"
git -C "$teammate_source" apply "$study/source.patch"
git -C "$teammate_source" apply --check "$study/quantization-config.patch"
git -C "$teammate_source" apply "$study/quantization-config.patch"
git -C "$teammate_source" apply --check "$study/ternary-model.patch"
git -C "$teammate_source" apply "$study/ternary-model.patch"
mkdir -p "$artifacts"
```

按 [数据校验命令](TRAINING.md#数据校验) 生成 `$artifacts/data-manifest.json`。只使用原来的一个 TUAR 数据集和按录音划分的 train/val/test，窗口数为 55,524 / 6,426 / 7,211；不重划数据。每次训练、校准、评估使用新的输出目录。

## INT4 与混合 W/A

统一传入 `--weight-bit-width 4 --activation-bit-width 4` 即 W4A4；换成 `4/8` 或 `8/4` 可分别检查权重或激活降位宽。训练和校准省略这些参数就是 W8A8；scheme 参数为 `--weight-quant-scheme uniform --activation-quant-scheme uniform`。

**PTQ：** 复用同 seed 的 FP32 微调检查点。单档校准与七档 sweep 二选一；单档示例采用原始 99.999 百分位。

```bash
fp32_checkpoint=/path/to/selected-fp32-finetune.ckpt
python "$study/calibrate.py" --source "$teammate_source" --data "$data" \
  --checkpoint "$fp32_checkpoint" --output "$artifacts/ptq-w4a4-single" \
  --weight-bit-width 4 --activation-bit-width 4 --percentile 99.999

python "$study/sweep.py" --source "$teammate_source" --data "$data" \
  --fp32 "$fp32_checkpoint" --output "$artifacts/ptq-w4a4-sweep" \
  --weight-bit-width 4 --activation-bit-width 4
```

校准固定 train 中同一份 seed42 的 512 窗口索引（16 个 batch32），保存索引清单和哈希。七档为 99.999/99.99/99.9/99/98/95/90，只用验证集选档位，再锁定 BAcc 阈值，最后测完整测试集一次。W4A4 检查点名为 `ptq_w4a4.ckpt`；默认 W8A8 保留 `ptq_int8.ckpt`。复用已有校准档位时，父检查点和 W/A 配置必须一致。

**C：** 复用同 seed 的 FP32 **预训练**父模型，仅做量化微调 5 轮、8,675 步，batch32、原生 `val_loss` 选检查点。已有父模型不重跑预训练。

```bash
seed=43
parent=/path/to/seed43/fp32/pretrain/complete.json
pretrained=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint"])' "$parent")
pretrained_sha=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint_sha256"])' "$parent")
python "$study/train_stage.py" --source "$teammate_source" --data-root "$data" \
  --data-manifest "$artifacts/data-manifest.json" --output "$artifacts/c-w4a4/finetune" \
  --group c --stage finetune --seed "$seed" \
  --pretrained "$pretrained" --pretrained-sha256 "$pretrained_sha" \
  --weight-bit-width 4 --activation-bit-width 4
```

**QAP B：** 预训练与微调必须使用相同 W/A 配置。预训练 10 轮、15,610 步，微调 5 轮、8,675 步，均 batch32。

```bash
seed=43
run="$artifacts/qap-b-w4a4"
python "$study/train_stage.py" --source "$teammate_source" --data-root "$data" \
  --data-manifest "$artifacts/data-manifest.json" --output "$run/pretrain" \
  --group qap_b --stage pretrain --seed "$seed" \
  --weight-bit-width 4 --activation-bit-width 4

pretrained=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint"])' "$run/pretrain/complete.json")
pretrained_sha=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint_sha256"])' "$run/pretrain/complete.json")
python "$study/train_stage.py" --source "$teammate_source" --data-root "$data" \
  --data-manifest "$artifacts/data-manifest.json" --output "$run/finetune" \
  --group qap_b --stage finetune --seed "$seed" \
  --pretrained "$pretrained" --pretrained-sha256 "$pretrained_sha" \
  --weight-bit-width 4 --activation-bit-width 4
```

`--plan-only` 可先解析训练配置。C 的 `pretrained_handoff.json` 记录 34 个共享 observer 加载后仍 fresh；`first_training_batch.json` 检查分类器全部 35 个 observer 的首次更新。

**评估：** 从微调完成记录读取选定检查点。未指定 W/A 字段时从检查点推断；指定的字段成为期望值，任何不匹配都会拒绝。下面以 C W4A4 为例；QAP B 使用其输出目录和 `--model-kind qap_b`。

```bash
complete="$artifacts/c-w4a4/finetune/complete.json"
checkpoint=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint"])' "$complete")
checkpoint_sha=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint_sha256"])' "$complete")
python "$study/evaluate_one.py" --source "$teammate_source" --data-root "$data" \
  --checkpoint "$checkpoint" --checkpoint-sha256 "$checkpoint_sha" \
  --model-kind c --seed "$seed" --output "$artifacts/c-w4a4/evaluation" \
  --weight-bit-width 4 --activation-bit-width 4
```

## 三值权重（已接入模型）

`ternary-model.patch` 把三值权重接进了全模型：`--weight-bit-width 2 --weight-quant-scheme ternary`（激活保持 8 位）。每个权重张量按 TWN 规则量化：scale 取该张量权重绝对值的平均值，`|w| > 0.7 × scale` 的权重映射为 ±1，其余为 0，重建值为 code × scale；梯度走 Brevitas 的三值直通估计器。覆盖范围和 uniform 相同（分类器 18 个权重张量，预训练 17 个），4 个 depthwise 卷积权重、归一化和 decoder 仍为 FP32。配置检查要求三值必须配 `weight_bit_width=2`，其他位宽会被拒绝；`test_ternary_model.py` 核对 18 个量化器都是 2 位、三个取值、scale 等于平均绝对值。训练、校准、评估命令与 INT4 相同，把位宽参数换成 `--weight-bit-width 2 --weight-quant-scheme ternary` 即可。

## 三值映射接口（独立实现，供导出与核对）

`models.femba_ternary` 另外提供一个不依赖模型的映射接口，和上面的模型量化器使用同一套 codes/threshold/scale 语义，用于导出与核对。full-range uniform 2-bit 有四个量化级别；Brevitas 默认权重使用 narrow range，2-bit 时虽有三个代码，也没有因此实现下述显式 threshold/scale/STE 契约。不能把 uniform 2-bit 配置改名为三值。

给定有限浮点输入 `x`、正的常量 `scale`、非负 `threshold`：`x < -threshold*scale` 映射为 −1，`x > threshold*scale` 映射为 +1，其余映射为 0，两个边界均归零。重建值为 `code*scale`；对输入使用 identity STE，scale 不学习。scale/threshold 的选择方法及使用的训练窗口必须记录，仅从训练数据决定并在验证、测试前冻结。

```bash
PYTHONPATH="$teammate_source" python - <<'PY'
import json
import torch
from models.femba_ternary import TernaryMapping, ternary_codes, ternary_fake_quantize

mapping = TernaryMapping(scale=2.0, threshold=0.5)
x = torch.tensor([-2., -1., 0., 1., 2.], requires_grad=True)
assert ternary_codes(x, mapping).tolist() == [-1, 0, 0, 0, 1]
y = ternary_fake_quantize(x, mapping)
assert y.tolist() == [-2., 0., 0., 0., 2.]
y.sum().backward()
assert torch.equal(x.grad, torch.ones_like(x))
serialized = json.dumps(mapping.to_dict(), allow_nan=False)
assert TernaryMapping.from_dict(json.loads(serialized)) == mapping
print(serialized)
PY
```

序列化契约完整保存 `mapping="symmetric_ternary"`、`version=1`、`codes=[-1,0,1]`、`scale`、`threshold`、`threshold_units="scale"`、`zero_boundary="inclusive"`、`ste="identity"`。代码返回 int8 容器中的三个值，不做位打包。后续模型接入应逐个声明权重/激活映射位置，保留未覆盖的 FP32 参数，并实现检查点一致性和严格重载；不能只改 scheme 名称来宣称已完成三值训练。

## 共用结果规范

每个 seed、方法、W/A 组合保留一份独立结果，至少包含：

- **配置与覆盖：** W/A 位宽、各自 scheme、enabled 状态、实际 proxy 位宽及计数、FP32 参数范围；三值另存上述完整映射配置和选择方法。
- **来源：** 上游 base commit、实际 dirty 状态、执行源码和 runner 哈希、历史与增量 patch 各自哈希、seed、父/最终检查点 SHA256、同初始化证明。不能只用 base commit 标识 dirty 源码。
- **数据与预算：** TUAR 及按录音划分说明、三个 H5 和划分清单哈希、窗口/类别数、预训练/微调 epoch 与真实 global step、batch、检查点选择指标；PTQ 另存 512 窗口索引清单哈希、百分位、验证选档及阈值锁定记录。
- **完整指标与核验：** 用 `float64(z1)-float64(z0)` 计算 AUROC/AP；同时记录默认与验证阈值下 Accuracy、BAcc、`[[TN,FP],[FN,TP]]`、预测正类比例。记录严格重载、PTQ 浮点参数未改、53 个 scale 一致和独立指标重算结果。
- **汇总：** 每 seed 数值、三 seed 均值和样本标准差（`ddof=1`），相同 seed 的差值及其均值/样本标准差。历史 QAP42 为队友检查点的 WSL 复测，单列来源；严格同初始化的 QAP B−C 比较只用 43/44，全三 seed 另列。

历史 [c-seed-results.json](c-seed-results.json) 保持原 W8A8 证据。低位宽结果另存，不能拿旧 JSON 改标签。公开材料只保留必要脚本、配置、归一化结果和曲线；数据、权重、逐样本预测、原始日志与工作记录留在实验目录。

## 验证记录

[配置契约验收](quantization-checks.json) 保存两份 patch、实际源码与 runner 的 SHA256，以及可复跑命令。CPU 套件共 68 项：67 通过，1 项旧 TensorBoard 事件导出测试因环境未安装 TensorBoard 而跳过；本次位宽、检查点、PTQ 和三值契约测试全部通过。

全尺寸模型检查默认 W8 与旧实现的状态、输出和 RNG；PTQ 重载使用缩短序列的合成 fixture，另行核对真实 512 索引清单及 16 个 batch 的顺序。当前未增加 INT4 / 三值训练或全测试集精度结果。
