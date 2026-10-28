"""Render mean training/validation losses from the portable CSV (CPU only)."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
COLORS = {'fp32': '#2364a4', 'c_w8a8': '#22846a', 'qap_w8a8': '#c66c20'}
LABELS = {'fp32': 'FP32', 'c_w8a8': 'C W8A8', 'qap_w8a8': 'QAP W8A8'}


def main() -> None:
    values = defaultdict(list)
    with (ROOT / 'learning-curves.csv').open() as handle:
        for row in csv.DictReader(handle):
            _, arm, stage = row['stage'].split('/')
            task = 'pretrain' if stage == 'pretrain' else ('tusl-v2' if 'tusl' in stage else stage.removeprefix('finetune-'))
            for loss in ('train_loss', 'val_loss'):
                values[(task, arm, loss, int(row['epoch']) + 1)].append(float(row[loss]))
    fig, axes = plt.subplots(4, 2, figsize=(10, 11), layout='constrained')
    for index, task in enumerate(('pretrain', 'tuar', 'tuab', 'tusl-v2')):
        for column, loss in enumerate(('train_loss', 'val_loss')):
            ax = axes[index, column]
            for arm in COLORS:
                epochs = sorted(epoch for t, a, l, epoch in values if (t, a, l) == (task, arm, loss))
                if epochs:
                    means = [sum(values[task, arm, loss, e]) / len(values[task, arm, loss, e]) for e in epochs]
                    ax.plot(epochs, means, color=COLORS[arm], label=LABELS[arm], linewidth=1.8)
            ax.set(title=f'{task.upper()} — {"training" if column == 0 else "validation"}', xlabel='Epoch (1-based)', ylabel='Loss')
            ax.grid(alpha=0.2)
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle('FEMBA formal FP32 / INT8 learning curves\nMeans across three seeds; TUSL also averages four folds', fontsize=13)
    fig.savefig(ROOT / 'learning-curves.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
