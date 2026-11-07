# 共用位宽配置与三值权重接口

> **正式线（按受试者划分，TUAB 预训练 2 轮，覆盖 INT8 / INT4 / 三值）按 [正式线运行说明](../formal_line/README.md) 运行**；本页说明底层补丁参数、单阶段命令行接口与三值权重实现（命令示例以 TUAR 数据目录为例）。

权重与激活位宽分别通过 `weight_bit_width` 和 `activation_bit_width` 控制（默认均为 8，支持 2–16 的整数）；量化方案通过 `weight_quant_scheme` 和 `activation_quant_scheme` 控制（默认为 `uniform`，权重另支持 `ternary` 三值量化）。

**量化覆盖范围**：分类模型共 **35 个激活量化器、18 个权重量化器**（预训练编码器为 **34 个激活、17 个权重**）；4 个 depthwise 卷积权重、归一化层与预训练解码器保持 FP32，`dt_rank=96`。量化配置写入检查点；加载已有量化检查点时会校验位宽与方案是否匹配，而加载 FP32 预训练检查点进入 C 组微调时，激活量化器会从头统计尺度。

## 独立准备源码

检出上游仓库 `a2a4d44`，依次应用三个补丁：[source.patch](source.patch)、[quantization-config.patch](quantization-config.patch)、[ternary-model.patch](ternary-model.patch)（Brevitas 版本固定为 0.12.1）：

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

按 [数据校验命令](TRAINING.md#数据校验) 生成 `$artifacts/data-manifest.json`。每次训练、校准、评估使用独立的输出目录。

## INT4 与混合位宽命令

传入 `--weight-bit-width 4 --activation-bit-width 8` 即为正式线的 INT4（W4A8）；传入 `4 / 4` 或 `8 / 4` 可测试激活降位宽。下面的示例都是 W4A4，跑正式线 INT4 时把 `--activation-bit-width 4` 换成 `8`；省略位宽参数则默认为 W8A8。

**1. PTQ（训练后量化）**：从同种子的 FP32 微调检查点出发，支持单档百分位校准（`calibrate.py`）或七档百分位扫描（`sweep.py`）：

```bash
fp32_checkpoint=/path/to/selected-fp32-finetune.ckpt
# 单档校准
python "$study/calibrate.py" --source "$teammate_source" --data "$data" \
  --checkpoint "$fp32_checkpoint" --output "$artifacts/ptq-w4a4-single" \
  --weight-bit-width 4 --activation-bit-width 4 --percentile 99.999

# 七档扫描（99.999 / 99.99 / 99.9 / 99 / 98 / 95 / 90），按验证集 AUROC 选档并评一次测试集
python "$study/sweep.py" --source "$teammate_source" --data "$data" \
  --fp32 "$fp32_checkpoint" --output "$artifacts/ptq-w4a4-sweep" \
  --weight-bit-width 4 --activation-bit-width 4
```

校准固定抽取训练集中的 512 个片段（种子 42，16 个 batch32），并保存样本索引清单与哈希。

**2. C 组（FP32 预训练 + 量化微调）**：复用同种子的 FP32 预训练检查点，只跑量化微调：

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

**3. QAP（量化预训练 + 量化微调）**：两阶段使用相同的位宽配置：

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

加上 `--plan-only` 可在不启动训练的情况下打印解析后的配置。

**4. 独立评估**：从微调目录的 `complete.json` 读取最佳检查点，先在验证集上确定最佳 BAcc 阈值，再评测试集：

```bash
complete="$artifacts/c-w4a4/finetune/complete.json"
checkpoint=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint"])' "$complete")
checkpoint_sha=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint_sha256"])' "$complete")
python "$study/evaluate_one.py" --source "$teammate_source" --data-root "$data" \
  --checkpoint "$checkpoint" --checkpoint-sha256 "$checkpoint_sha" \
  --model-kind c --seed "$seed" --output "$artifacts/c-w4a4/evaluation" \
  --weight-bit-width 4 --activation-bit-width 4
```

## 三值权重（Ternary）

`ternary-model.patch` 将三值权重量化接入全模型：传入 `--weight-bit-width 2 --weight-quant-scheme ternary`（激活保持 8 位）。

每个量化权重张量按三值权重网络（TWN）规则计算：
- 缩放因子 `scale` 取该张量权重绝对值的均值 `mean(|w|)`；
- `|w| > 0.7 × scale` 映射为 `±1`，其余映射为 `0`，反量化值为 `code × scale`；
- 反向传播使用直通估计器（STE，前向离散化、反向直接传梯度）。

量化层范围与 uniform 相同（分类器 18 个权重张量，预训练 17 个），配置检查要求 `ternary` 必须搭配 `weight_bit_width=2`。训练、校准与评估命令只需把位宽参数换成 `--weight-bit-width 2 --weight-quant-scheme ternary`。

## 独立三值映射模块（`models.femba_ternary`）

`models.femba_ternary` 提供一个不依赖完整模型的三值量化辅助模块，使用相同的阈值与缩放语义，供单元测试与权重导出核对：

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

## 结果记录字段与单元测试

每个阶段的完成记录与评估输出均保存：
- **量化配置**：权重与激活位宽、量化方案、覆盖层数与 FP32 保留参数；
- **来源哈希**：上游提交号、补丁文件哈希、运行脚本哈希、数据清单哈希、父检查点与最佳检查点 SHA256；
- **评测指标**：测试集 AUROC、AP，以及默认阈值（0）和验证集选定阈值下的准确率、平衡准确率（BAcc）、混淆矩阵与判正类比例。

CPU 单元测试覆盖位宽配置校验、检查点保存与重载、PTQ 校准与三值映射（详见 [quantization-checks.json](quantization-checks.json)）。
