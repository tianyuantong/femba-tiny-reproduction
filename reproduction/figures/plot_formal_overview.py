"""Draw the front-page charts, test AUROC of the formal line at INT8 and INT4, from the summaries (light and dark SVG)."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).parent
RESULTS = ROOT.parent / "formal_line" / "results"
SEEDS = ("42", "43", "44")
SETTINGS = (("int8", "w8a8", "INT8", "W8A8", "formal-overview"), ("int4", "w4a8", "INT4", "W4A8", "formal-overview-int4"))
TASKS = (("tuar", "auroc", "TUAR 伪迹检测"), ("tuab", "auroc", "TUAB 异常检测"), ("tusl-v2", "macro_auroc", "TUSL v2 慢波三分类"))
ARMS = (("fp32", "FP32 基线"), ("ptq", "PTQ（训练后量化）"), ("c", "QAT（FP32 预训练 + 量化微调）"), ("qap", "QAP（量化预训练 + 量化微调）"))
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e1e0d9",
              "fp32": "#6250d6", "ptq": "#eb6834", "c": "#2a78d6", "qap": "#1baf7a"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "muted": "#c3c2b7", "grid": "#2c2c2a",
             "fp32": "#9085e9", "ptq": "#d95926", "c": "#3987e5", "qap": "#199e70"},
}
FONT = "-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif"
WIDTH, HEIGHT, LEFT, RIGHT = 760, 410, 52, 736
TOP, BOTTOM, LOW, HIGH = 108, 360, 0.5, 0.9  # the axis starts at chance level
BAR, GAP = 24, 6


def y(value: float) -> float:
    return BOTTOM - (value - LOW) / (HIGH - LOW) * (BOTTOM - TOP)


def text(x, top, body, size, fill, anchor="start", weight=None):
    bold = f" font-weight='{weight}'" if weight else ""
    return (f"<text x='{x}' y='{top}' font-size='{size}' fill='{fill}' text-anchor='{anchor}'{bold} "
            f"font-family='{FONT}'>{body}</text>")


def draw(theme: dict, summary: dict, name: str, config: str) -> str:
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {WIDTH} {HEIGHT}' width='{WIDTH}' height='{HEIGHT}' role='img'>",
             f"<title>FEMBA-Tiny formal line: {name} test AUROC on three tasks, three seeds</title>",
             f"<rect width='{WIDTH}' height='{HEIGHT}' fill='{theme['surface']}' rx='8'/>",
             text(LEFT, 28, f"三任务 {name}（{config}）测试 AUROC", 17, theme["ink"], weight=600),
             text(LEFT, 47, "FEMBA-Tiny · 受试者独立划分 · 3 个随机种子的均值，细线为范围 · 纵轴从随机水平 0.5 起", 11.5, theme["muted"])]
    x = LEFT
    for arm, label in ARMS:
        parts.append(f"<rect x='{x}' y='61' width='12' height='12' rx='3' fill='{theme[arm]}'/>")
        parts.append(text(x + 17, 71.5, label, 11.5, theme["ink"]))
        x += 17 + 11.5 * len(label) * 0.86 + 18
    for tick in (0.5, 0.6, 0.7, 0.8, 0.9):
        parts.append(f"<line x1='{LEFT}' y1='{y(tick):.1f}' x2='{RIGHT}' y2='{y(tick):.1f}' stroke='{theme['grid']}' stroke-width='1'/>")
        parts.append(text(LEFT - 8, round(y(tick) + 4, 1), f"{tick:.2f}", 10.5, theme["muted"], "end"))
    band = (RIGHT - LEFT) / len(TASKS)
    group = len(ARMS) * BAR + (len(ARMS) - 1) * GAP
    for index, (task, metric, label) in enumerate(TASKS):
        centre = LEFT + band * (index + 0.5)
        for position, (arm, _) in enumerate(ARMS):
            values = [summary[task][arm][seed][metric] for seed in SEEDS]
            mean, left = statistics.mean(values), centre - group / 2 + position * (BAR + GAP)
            top, mid = y(mean), left + BAR / 2
            parts.append(f"<path d='M{left},{BOTTOM} V{top + 4:.1f} Q{left},{top:.1f} {left + 4},{top:.1f} "
                         f"H{left + BAR - 4} Q{left + BAR},{top:.1f} {left + BAR},{top + 4:.1f} V{BOTTOM} Z' fill='{theme[arm]}'/>")
            high, low = y(max(values)), y(min(values))
            parts.append(f"<line x1='{mid}' y1='{high:.1f}' x2='{mid}' y2='{low:.1f}' stroke='{theme['muted']}' stroke-width='1.2'/>")
            parts.append(text(mid, round(high - 6, 1), f"{mean:.2f}", 10.5, theme["ink"], "middle", 600))
        parts.append(text(round(centre, 1), BOTTOM + 22, label, 11.5, theme["ink"], "middle", 600))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    for folder, setting, name, config, stem in SETTINGS:
        summary = json.loads((RESULTS / folder / "summary.json").read_text())[setting]
        for mode, theme in THEMES.items():
            (ROOT / f"{stem}-{mode}.svg").write_text(draw(theme, summary, name, config))


if __name__ == "__main__":
    main()
