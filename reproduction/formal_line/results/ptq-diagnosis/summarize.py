"""Recompute the numbers in README.md from the result records in this directory.

Reads calibration/, ablation/, sensitivity/, sweep-scales.json and the TUAB FP32 rows of
../int8/summary.json, and prints one block per README table (test AUROC unless marked
validation). Standard library only. Usage: python summarize.py
"""
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)
SETTINGS = ("w8a8", "w4a8", "ternary")
ROLES = ("all", "weights", "activations")
TAIL_TOP = 8
SCAN_OUTPUT = "mamba_blocks.1.mamba_rev.ssm.output_quant.act_quant"


@dataclass(frozen=True)
class Records:
    fp32: list[float]
    calibration: dict  # seed -> (clipping, quantized role) -> variant
    ablation: dict     # (seed, setting) -> record
    split: dict        # (seed, setting) -> variant name -> variant
    sensitivity: dict  # seed -> record
    scales: dict       # seed -> percentile -> checkpoint and activation scales


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def load_records() -> Records:
    ablation = {(seed, setting): load(f"ablation/seed{seed}-{setting}.json") for seed in SEEDS for setting in SETTINGS}
    return Records(
        fp32=[load("../int8/summary.json")["w8a8"]["tuab"]["fp32"][str(seed)]["auroc"] for seed in SEEDS],
        calibration={seed: {(v["calibration"], v["quantized"]): v for v in load(f"calibration/seed{seed}.json")["variants"]}
                     for seed in SEEDS},
        ablation=ablation,
        split={key: {v["variant"]: v for v in record["variants"]} for key, record in ablation.items()},
        sensitivity={seed: load(f"sensitivity/seed{seed}.json") for seed in SEEDS},
        scales=load("sweep-scales.json"))


def short(name: str) -> str:
    return name.removeprefix("mamba_blocks.").removesuffix(".act_quant").replace("mamba_", "")


def row(label: str, values: list[float], digits: int = 4, should_add_mean: bool = True) -> str:
    shown = [*values, mean(values)] if should_add_mean else values
    return f"| {label} | " + " | ".join(f"{v:.{digits}f}" for v in shown) + " |"


def auroc(variant: dict) -> float:
    return variant["test"]["auroc"]


def print_overview(records: Records) -> None:
    print("Overview: seed 42 | seed 43 | seed 44 | mean")
    print(row("FP32", records.fp32))
    for role in ROLES:
        print(row(f"W8A8 {role}", [auroc(records.split[seed, "w8a8"][role]) for seed in SEEDS]))
    fp16 = [auroc(records.sensitivity[seed]["fp16_variant"]) for seed in SEEDS]
    print(row("FP16 fallback", fp16))
    every, weights = ([auroc(records.split[seed, "w8a8"][role]) for seed in SEEDS] for role in ("all", "weights"))
    print(f"recovered {(mean(fp16) - mean(every)) / (mean(weights) - mean(every)):.1%} of the loss; "
          f"test gap to weights only {[round(w - f, 4) for w, f in zip(weights, fp16)]}, mean {mean(weights) - mean(fp16):.4f}")


def print_clipping_and_bit_widths(records: Records) -> None:
    print("\nActivation clipping, W8A8")
    for method in ("percentile", "mse", "kl"):
        print(row(method, [auroc(records.calibration[seed][method, "all"]) for seed in SEEDS]))
    print("\nBit widths: all | weights only | activations only, then FP32 minus each (means)")
    for setting in SETTINGS:
        means = [mean(auroc(records.split[seed, setting][role]) for seed in SEEDS) for role in ROLES]
        print(f"| {setting} | " + " | ".join(f"{v:.4f}" for v in means) + " | "
              + " | ".join(f"{mean(records.fp32) - v:.4f}" for v in means) + " |")
    print("ternary weights only per seed", [round(auroc(records.split[seed, "ternary"]["weights"]), 4) for seed in SEEDS])
    print("activations only identical across settings:",
          all(auroc(records.split[seed, setting]["activations"]) == auroc(records.split[seed, "w8a8"]["activations"])
              for seed in SEEDS for setting in SETTINGS))
    for k in (2, 4, TAIL_TOP):
        print(row(f"W8A8, heaviest-tail top {k} in float",
                  [auroc(records.split[seed, "w8a8"][f"float_top{k}"]) for seed in SEEDS]))


def print_sensitivity(records: Records) -> None:
    print("\nSensitivity (validation): top three by AUROC drop when quantized alone, worst single drop per block")
    for seed in SEEDS:
        record = records.sensitivity[seed]
        drop = {name: record["reference_validation_auroc"]["weights_only"] - value
                for name, value in record["alone_validation_auroc"].items()}
        top = ", ".join(f"{short(name)} {drop[name]:.3f}" for name in record["ranking"][:3])
        worst = {block: max(d for name, d in drop.items() if name.startswith(f"mamba_blocks.{block}.")) for block in (0, 1)}
        print(f"| {seed} | {top} | block 0 {worst[0]:.4f}, block 1 {worst[1]:.4f} |")

    print("\nFP16 fallback: k | FP16 share | test AUROC | heaviest-tail top 8 in float: test AUROC | share")
    for seed in SEEDS:
        record = records.sensitivity[seed]
        elements = record["elements_per_window"]
        tail = records.ablation[seed, "w8a8"]["tail_ranking"][:TAIL_TOP]
        print(f"| {seed} | {record['chosen_k']} | {record['fp16_element_share']:.1%} | {auroc(record['fp16_variant']):.4f} | "
              f"{auroc(records.split[seed, 'w8a8'][f'float_top{TAIL_TOP}']):.4f} | "
              f"{sum(elements[name] for name in tail) / sum(elements.values()):.1%} |")
        print("   ", [short(name) for name in record["fp16_quantizers"]])
    print(f"mean FP16 share {mean(records.sensitivity[seed]['fp16_element_share'] for seed in SEEDS):.1%}")

    print("\nValidation AUROC: all INT8 | top k in FP16 for k = "
          + ", ".join(records.sensitivity[SEEDS[0]]["fp16_fallback_validation_auroc"]) + " | weights only")
    for seed in SEEDS:
        reference = records.sensitivity[seed]["reference_validation_auroc"]
        print(row(str(seed), [reference["all"], *records.sensitivity[seed]["fp16_fallback_validation_auroc"].values(),
                              reference["weights_only"]], digits=3, should_add_mean=False))


def print_tails(records: Records) -> None:
    print("\nTails: 99.999 / 90 percentile scale; top 7 range | their sensitivity ranks | 8th | ratio | rank")
    for seed in SEEDS:
        per = records.scales[str(seed)]
        ratio = {name: per["99.999"]["activation_scales"][name] / per["90"]["activation_scales"][name]
                 for name in per["90"]["activation_scales"]}
        tails = records.ablation[seed, "w8a8"]["tail_ranking"]
        assert tails == sorted(ratio, key=ratio.get, reverse=True)
        rank = {name: index + 1 for index, name in enumerate(records.sensitivity[seed]["ranking"])}
        first, eighth = tails[:TAIL_TOP - 1], tails[TAIL_TOP - 1]
        print(f"| {seed} | {min(ratio[n] for n in first):.3g}–{max(ratio[n] for n in first):.3g} | "
              f"{min(rank[n] for n in first)}–{max(rank[n] for n in first)} | {short(eighth)} | {ratio[eighth]:.3g} | {rank[eighth]} |")

    print(f"\n{short(SCAN_OUTPUT)} scale over its 95 percentile scale, and the largest such ratio in block 0")
    for seed in SEEDS:
        per = records.scales[str(seed)]
        over = {pct: per[pct]["activation_scales"][SCAN_OUTPUT] / per["95"]["activation_scales"][SCAN_OUTPUT]
                for pct in ("99.99", "99.999")}
        block0 = max(per["99.999"]["activation_scales"][name] / per["95"]["activation_scales"][name]
                     for name in per["95"]["activation_scales"] if name.startswith("mamba_blocks.0."))
        print(f"| {seed} | 99.99: {over['99.99']:.0f} | 99.999: {over['99.999']:.0f} | block 0 max: {block0:.3g} |")


def main() -> None:
    records = load_records()
    print_overview(records)
    print_clipping_and_bit_widths(records)
    print_sensitivity(records)
    print_tails(records)


if __name__ == "__main__":
    main()
