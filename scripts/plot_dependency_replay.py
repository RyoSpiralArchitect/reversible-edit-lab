"""Export a static, evidence-bound overview of component reuse and measured cost."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    summary = json.loads((args.run / 'summary.json').read_text())
    modes = ['fresh', 'old_target', 'old_keys', 'old_both', 'old_update', 'checked']
    labels = ['Recompute all', 'Reuse old target', 'Reuse old keys / solve', 'Reuse target + keys', 'Reuse whole update', 'Check dependencies']
    matrix = np.array([[summary['comparisons'][a][m]['state_matches'] / summary['comparisons'][a][m]['cells']
                        for a in ['rome', 'memit']] for m in modes])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={'width_ratios': [1, 1.4]})
    fig.patch.set_facecolor('#fafafa')
    axes[0].imshow(matrix, cmap=matplotlib.colors.LinearSegmentedColormap.from_list('matches', ['#e8e8ef', '#8bcdbb']), vmin=0, vmax=1)
    axes[0].set_xticks([0, 1], ['ROME', 'MEMIT'])
    axes[0].set_yticks(range(6), labels)
    for i in range(6):
        for j, algorithm in enumerate(['rome', 'memit']):
            cell = summary['comparisons'][algorithm][modes[i]]
            axes[0].text(j, i, f"{cell['state_matches']}/{cell['cells']}", ha='center', va='center', fontsize=13)
    axes[0].set_title('Full-state matches to fresh B-only', fontsize=12, pad=14)
    ax = axes[1]
    ticks = []
    for i, row in enumerate(summary['timings']):
        ratios = row['paired_ratios']
        ax.scatter(ratios, [i] * len(ratios), color='#394a73', alpha=.6, s=24)
        ax.plot([min(ratios), max(ratios)], [i, i], color='#394a73', alpha=.4)
        ax.scatter([row['paired_ratio_median']], [i], color='#db7446', marker='D', s=40, zorder=3)
        ticks.append(row['algorithm'].upper() + ' / ' + row['pair'].replace('_', ' '))
    ax.axvline(1, linestyle='--', color='#777777', linewidth=1)
    ax.set_yticks(range(len(ticks)), ticks)
    ax.invert_yaxis()
    ax.set_xlabel('Checked reuse / author recomputation time')
    ax.set_title('Cost includes dependency checks', fontsize=12, pad=14)
    ax.grid(axis='x', alpha=.15)
    ax.set_xlim(left=.95)
    for axis in axes:
        for spine in axis.spines.values():
            spine.set_visible(False)
    histories = summary['histories']
    matches = sum(r['arms']['checked']['state_bitwise'] for r in histories)
    fig.suptitle('Component reuse after withdrawing a weight edit', fontsize=17, y=.99)
    fig.text(.5, .055, f"8/16-edit histories: checked replay matches {matches}/{len(histories)} fresh retained-history references.", ha='center', fontsize=11)
    fig.text(.5, .015, 'Pinned GPT-2 CPU FP32. Three paired timing repeats per cell; dots = observations, diamond = median. >1 is slower.', ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, .1, 1, .94), w_pad=3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(args.out.with_suffix('.pdf'), facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == '__main__':
    main()
