"""Draw where INT8 activation quantization costs PTQ and QAP validation AUROC on TUAB (light and dark SVG)."""
import json
from pathlib import Path
import statistics

from plot_formal_overview import FONT, THEMES, text

ROOT = Path(__file__).parent
RESULTS = ROOT.parent / "formal_line" / "results" / "layers"
SEEDS = (42, 43, 44)
MODELS = (("ptq", "PTQ（训练后量化）"), ("qap", "QAP（量化预训练 + 量化微调）"))
GROUPS = (("other", "前后端"), ("block0", "第一块"), ("block1", "第二块"))
SITES = (("mamba_rev.ssm", "反向扫描输出"), ("mamba_rev.silu", "反向 SiLU"), ("mamba_fwd.ssm", "正向扫描输出"),
         ("mamba_rev.conv1d", "反向卷积"), ("mamba_rev.in_proj_quant", "反向输入投影"))
WIDTH, HEIGHT, LEFT, SPLIT, RIGHT = 760, 404, 52, 330, 736
TOP, BOTTOM, LOW, HIGH = 126, 366, -0.04, 0.10
BAR, GAP = 20, 12


def y(value: float) -> float:
    return BOTTOM - (value - LOW) / (HIGH - LOW) * (BOTTOM - TOP)


def site_name(suffix: str) -> str:
    return f"mamba_blocks.1.{suffix}.act_quant" if suffix.endswith("_quant") else f"mamba_blocks.1.{suffix}.output_quant.act_quant"


def label_value(mean: float) -> str:
    return "0.000" if abs(mean) < 0.0005 else f"{mean:+.3f}".replace("-", "−")


def drops(results: dict, model: str, section: str, key: str) -> list[float]:
    return [results[seed][model]["weights_only"]["auroc"] - results[seed][model][section][key]["auroc"] for seed in SEEDS]


def bars(theme: dict, centre: float, label: str, rows: dict) -> list[str]:
    parts = [text(round(centre, 1), BOTTOM + 22, label, 11, theme["ink"], "middle", 600)]
    for position, (model, _) in enumerate(MODELS):
        values, left = rows[model], centre - BAR - GAP / 2 + position * (BAR + GAP)
        mean = statistics.mean(values)
        top, base = min(y(mean), y(0)), max(y(mean), y(0))
        parts.append(f"<rect x='{left:.1f}' y='{top:.1f}' width='{BAR}' height='{max(base - top, 1):.1f}' rx='2' fill='{theme[model]}'/>")
        mid, high, low = left + BAR / 2, y(max(values)), y(min(values))
        parts.append(f"<line x1='{mid:.1f}' y1='{high:.1f}' x2='{mid:.1f}' y2='{low:.1f}' stroke='{theme['muted']}' stroke-width='1.2'/>")
        anchor = round(high - 6, 1) if mean >= 0 else round(low + 14, 1)
        parts.append(text(round(mid, 1), anchor, label_value(mean), 10, theme["ink"], "middle", 600))
    return parts


def draw(theme: dict, results: dict) -> str:
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {WIDTH} {HEIGHT}' width='{WIDTH}' height='{HEIGHT}' role='img'>",
             "<title>FEMBA-Tiny TUAB INT8: validation AUROC drop by activation group for PTQ and QAP</title>",
             f"<rect width='{WIDTH}' height='{HEIGHT}' fill='{theme['surface']}' rx='8'/>",
             text(LEFT, 28, "PTQ 的掉点集中在第二个 Mamba 块，QAP 在这里不掉点", 17, theme["ink"], weight=600),
             text(LEFT, 47, "TUAB · INT8 · 验证集 AUROC · 3 个随机种子的均值，细线为范围", 11.5, theme["muted"]),
             text(LEFT, 63, "掉点 = 只量化权重的 AUROC − 再打开这组激活量化的 AUROC；负值表示打开后变好", 11.5, theme["muted"])]
    x = LEFT
    for model, label in MODELS:
        parts.append(f"<rect x='{x}' y='75' width='12' height='12' rx='3' fill='{theme[model]}'/>")
        parts.append(text(x + 17, 85.5, label, 11.5, theme["ink"]))
        x += 17 + 11.5 * len(label) * 0.86 + 18
    parts.append(text(LEFT, 113, "三组激活量化，分别打开", 11.5, theme["muted"], weight=600))
    parts.append(text(SPLIT + 16, 113, "第二个 Mamba 块里掉点最多的五处，单独打开", 11.5, theme["muted"], weight=600))
    for tick in (-0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.10):
        stroke = theme["muted"] if tick == 0 else theme["grid"]
        parts.append(f"<line x1='{LEFT}' y1='{y(tick):.1f}' x2='{RIGHT}' y2='{y(tick):.1f}' stroke='{stroke}' stroke-width='1'/>")
        parts.append(text(LEFT - 8, round(y(tick) + 4, 1), f"{tick:+.2f}".replace("-", "−") if tick else "0", 10.5, theme["muted"], "end"))
    band = (SPLIT - LEFT) / len(GROUPS)
    for index, (key, label) in enumerate(GROUPS):
        rows = {model: drops(results, model, "groups", key) for model, _ in MODELS}
        parts += bars(theme, LEFT + band * (index + 0.5), label, rows)
    band = (RIGHT - SPLIT - 16) / len(SITES)
    for index, (suffix, label) in enumerate(SITES):
        rows = {model: drops(results, model, "alone", site_name(suffix)) for model, _ in MODELS}
        parts += bars(theme, SPLIT + 16 + band * (index + 0.5), label, rows)
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    results = {seed: json.loads((RESULTS / f"seed{seed}.json").read_text()) for seed in SEEDS}
    for name, theme in THEMES.items():
        (ROOT / f"layer-drop-{name}.svg").write_text(draw(theme, results))


if __name__ == "__main__":
    main()
