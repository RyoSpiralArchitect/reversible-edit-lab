"""Generate a readable report and optional static figures from raw evidence."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import numpy as np


LABELS = {"checkpoint": "Checkpoint", "delta_log": "Delta log", "factor": "Delta log",
          "old_response_log": "Old response", "analytic": "Analytic inverse",
          "svd_top1": "SVD top-1", "bottom_rank": "SVD top-1",
          "no_rollback": "No rollback", "random": "Random query", "uncertainty": "Uncertainty query"}
COLORS = {"checkpoint": "#374151", "delta_log": "#007f86", "factor": "#007f86",
          "old_response_log": "#82b440", "analytic": "#ae5ca8", "svd_top1": "#c77720",
          "bottom_rank": "#c77720", "no_rollback": "#bf4252", "random": "#5273b1", "uncertainty": "#568648"}


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def summarize(data: dict) -> dict:
    final_cycle = data["config"]["cycle_count"]
    grouped = defaultdict(list)
    for row in data["cycle_rows"]:
        if row["cycle"] == final_cycle:
            grouped[(row["precision"], row["method"])].append(row)
    cycles = []
    for (precision, method), rows in grouped.items():
        cycles.append({"precision": precision, "method": method, "n_seeds": len(rows),
                       "mean_weight_relative_error": _mean(rows, "weight_relative_error"),
                       "worst_weight_relative_error": max(r["weight_relative_error"] for r in rows),
                       "mean_probe_kl": _mean(rows, "probe_mean_kl"),
                       "mean_next_update_error": _mean(rows, "next_update_weight_relative_error")})
    active_groups = defaultdict(list)
    agreement_groups = defaultdict(list)
    for run in data["active_runs"]:
        precision = run["config"]["precision"]
        checkpoint_path = next(p for p in run["active_paths"]
                               if p["protocol"] == "checkpoint" and p["candidate_order"] == "forward")
        for path in run["active_paths"]:
            if path["candidate_order"] != "forward":
                continue
            active_groups[(precision, path["protocol"])].append(path)
            agreement_groups[(precision, path["protocol"])].append(
                path["selected_pool_ids"] == checkpoint_path["selected_pool_ids"])
        for path in run.get("query_baselines", []):
            active_groups[(precision, path["query_policy"])].append(path)
    active = []
    for (precision, method), rows in active_groups.items():
        agreements = agreement_groups.get((precision, method), [])
        active.append({"precision": precision, "method": method, "n_seeds": len(rows),
                       "mean_initial_accuracy": _mean(rows, "initial_test_accuracy"),
                       "mean_final_accuracy": _mean(rows, "final_test_accuracy"),
                       "min_final_accuracy": min(r["final_test_accuracy"] for r in rows),
                       "max_final_accuracy": max(r["final_test_accuracy"] for r in rows),
                       "mean_teacher_queries": _mean(rows, "teacher_label_queries"),
                       "mean_hypothetical_edits": _mean(rows, "hypothetical_edits"),
                       "selection_sequence_matches_checkpoint": int(sum(agreements)) if agreements else None})
    return {"cycle_count": final_cycle, "cycles": cycles, "active": active,
            "kernel_comparisons": len(data["kernel_rows"]), "cycle_observations": len(data["cycle_rows"])}


def write_report(data: dict, directory: Path, plots: bool = False) -> dict:
    summary = summarize(data)
    (directory / "summary.json").write_text(json.dumps(summary, allow_nan=False, indent=2), encoding="utf-8")
    seeds = ", ".join(map(str, data["config"]["seeds"]))
    lines = ["# Reversible Edit Lab — 実行結果", "",
             "## 今回測ったもの", "",
             "合成した線形連想記憶に、ROME/MEMITの単一行列・固定目標の更新を適用した結果。事前学習済みLLMの事実編集や、実機TTTの性能を測ったものではない。", "",
             f"- seeds: `{seeds}`", f"- 行列: 出力 {data['config']['output_dim']} × 入力 {data['config']['input_dim']}",
             f"- 単発比較: {summary['kernel_comparisons']}件; 反復の観測点: {summary['cycle_observations']}件",
             "- 行列演算・記録はfloat64。各重みの書き込み時に指定精度へ丸める。bfloat16は直接丸めをエミュレート。",
             "- SVDは事前に固定したtop-1除去。元重みやテストラベルによるランク選びは行わない。",
             "- 復元器へ渡す情報は方式ごとに限定。評価器は元行列を保持するので、実行プロセスのピークメモリ節約を示す測定ではない。", "",
             "## 反復したときの復元誤差", "",
             f"書き込み強度 `alpha=0.95` で編集→撤回を {summary['cycle_count']} 回。値はseed間の算術平均。相対Frobenius誤差は `||W_recovered-W_start|| / ||W_start||`。", "",
             "| 保存精度 | 復元方法 | 相対重み誤差 | probe KL | 次の同じ更新後の相対誤差 |",
             "| --- | --- | ---: | ---: | ---: |"]
    for row in summary["cycles"]:
        lines.append(f"| {row['precision']} | {LABELS[row['method']]} | {row['mean_weight_relative_error']:.4g} | {row['mean_probe_kl']:.4g} | {row['mean_next_update_error']:.4g} |")
    lines += ["", "## 情報の保存量", "",
              "`kernel.csv` は復元用数値ペイロードの実サイズを記録する。差分因子・内部目標・方向・共分散をfloat64で保存する費用を含む。Pythonオブジェクト、NPZの圧縮率、一時作業メモリは含めない。", "",
              "小さな行列では、MEMIT逆変換用の密な共分散がチェックポイントより大きくなる。『元の重みを保存しない』ことからメモリ節約を自動的に結論しない。", ""]
    if summary["active"]:
        lines += ["## 能動的な問い合わせのtoy実験", "",
                  "同じ初期分類器から、各候補ラベルを仮編集して未ラベルprobeの期待エントロピー減少を採点する。教師への問い合わせは選択確定後に1ラウンド1回だけ。テストラベルは採点専用。", "",
                  "以下は候補を順方向に評価した実行。チェックポイントとの経路一致は、全ラウンドの問い合わせ先列が一致したseed数。random/uncertaintyは同じラベル予算・本更新則だが計算量を一致させていない。3seed等の小規模結果から優位性の統計的主張はしない。", "",
                  "| 精度 | 方法 | 開始精度 | 最終精度 | 経路一致 / seeds | 仮編集数 |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"]
        for row in summary["active"]:
            matches = row["selection_sequence_matches_checkpoint"]
            agreement = "—" if matches is None else f"{matches}/{row['n_seeds']}"
            lines.append(f"| {row['precision']} | {LABELS[row['method']]} | {100*row['mean_initial_accuracy']:.2f}% | {100*row['mean_final_accuracy']:.2f}% | {agreement} | {row['mean_hypothetical_edits']:.0f} |")
        lines += ["", "復元の忠実さと選択方針の良さは別に評価する。期待エントロピー減少は自信の増加を測る代理目的で、正確さの改善を保証しない。`results.json` に逆順評価・seed別の全履歴・外部ラベル問い合わせ数を残した。", ""]
    scalar = data["selective_undo"]["scalar"]
    matrix = data["selective_undo"]["matrix"]
    lines += ["## 過去の一編集だけを引く場合", "",
              f"スカラー状態0をAで1、Bで2に上書きすると、Aの差分だけ引いた値は **{scalar['subtract_a_only']:g}**。AがなくBだけを学んだ値は **{scalar['b_without_a']:g}**。B→Aの順に戻してBを再計算した値は **{scalar['undo_b_a_then_replay_b']:g}**。", "",
              f"重なるキーを持つ行列例でも、単純な差分除去の相対誤差は `{matrix['naive']['weight_relative_error']:.6g}`、巻き戻して再適用した相対誤差は `{matrix['replayed']['weight_relative_error']:.6g}`。後続更新が先行更新に依存することを示す。", "",
              "## 再現と読み方", "",
              "- `results.json`: 全設定、seed別の生結果、環境情報、ソースのSHA-256。",
              "- `kernel.csv`: 単発の復元、書き込みの強さ、数値ペイロード量、次の更新。",
              "- `cycles.csv`: 編集と撤回の反復。",
              "- `summary.json`: 本ページの集計。",
              "- `../docs/METHODS.md` またはプロジェクト直下の `docs/METHODS.md`: 前提と比較条件。", "",
              "実行時間は参考値で、ハードウェア性能や全LLMの推論速度の比較には使用しない。", ""]
    if plots:
        figures = plot_figures(data, directory)
        lines += ["## 図", ""]
        for name in figures:
            lines.extend([f"![{name}]({name})", ""])
    (directory / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def plot_figures(data: dict, directory: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.18, "figure.facecolor": "#fafaf8",
                         "axes.facecolor": "#fafaf8", "savefig.facecolor": "#fafaf8"})
    precisions = [p for p in data["config"]["precisions"] if p != "float64"]
    figures = []
    if precisions:
        fig, axes = plt.subplots(1, len(precisions), figsize=(4.1*len(precisions), 3.7), squeeze=False)
        for ax, precision in zip(axes[0], precisions):
            for method in ("delta_log", "analytic"):
                selected = [r for r in data["kernel_rows"] if r['editor']=='rome' and r['precision']==precision
                            and r['method']==method and 0 < r['alpha'] < 1]
                alphas = sorted({r['alpha'] for r in selected})
                values = [max(_mean([r for r in selected if r['alpha']==a], 'weight_relative_error'), 1e-18) for a in alphas]
                ax.loglog([1-a for a in alphas], values, marker='o', label=LABELS[method], color=COLORS[method])
            ax.invert_xaxis()
            ax.set(title=precision, xlabel='Remaining old-response fraction (1 - alpha)', ylabel='Relative weight error after undo')
            ax.legend(frameon=False, fontsize=9)
        fig.suptitle('Strong writes amplify error in the analytic inverse', fontsize=14, y=1.02)
        fig.tight_layout()
        name='write_strength.png'; fig.savefig(directory/name, dpi=170, bbox_inches='tight'); plt.close(fig); figures.append(name)
    prec = data['config']['precisions']
    fig, axes = plt.subplots(1, len(prec), figsize=(3.7*len(prec), 3.8), squeeze=False)
    for ax, precision in zip(axes[0], prec):
        for method in ('delta_log','analytic','svd_top1'):
            selected=[r for r in data['cycle_rows'] if r['precision']==precision and r['method']==method]
            steps=sorted({r['cycle'] for r in selected})
            values=[max(_mean([r for r in selected if r['cycle']==s], 'weight_relative_error'),1e-18) for s in steps]
            ax.loglog(steps,values,marker='o',label=LABELS[method],color=COLORS[method])
        ax.set(title=precision,xlabel='Edit / undo cycles',ylabel='Relative error from initial weight')
        ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Repeated rollback: alpha = 0.95; mean across seeds',fontsize=14,y=1.02)
    fig.tight_layout(); name='repeated_rollback.png'; fig.savefig(directory/name,dpi=170,bbox_inches='tight'); plt.close(fig); figures.append(name)
    if data['active_runs']:
        summary=summarize(data)
        prec=[p for p in ('float32','bfloat16') if p in data['config']['precisions']] or [data['config']['precisions'][0]]
        fig, axes=plt.subplots(1,len(prec),figsize=(6.5*len(prec),4.5),squeeze=False)
        methods=('checkpoint','factor','analytic','bottom_rank','no_rollback','random','uncertainty')
        for ax,precision in zip(axes[0],prec):
            rows={r['method']:r for r in summary['active'] if r['precision']==precision}
            present=[m for m in methods if m in rows]
            means=[rows[m]['mean_final_accuracy'] for m in present]
            errors=np.array([[means[i]-rows[m]['min_final_accuracy'] for i,m in enumerate(present)],
                             [rows[m]['max_final_accuracy']-means[i] for i,m in enumerate(present)]])
            ax.bar(range(len(present)),means,color=[COLORS[m] for m in present],alpha=.88,yerr=errors,capsize=3)
            ax.axhline(rows[present[0]]['mean_initial_accuracy'],color='#444444',linestyle='--',label='Initial mean')
            ax.set_xticks(range(len(present)),[LABELS[m] for m in present],rotation=35,ha='right')
            ax.set(title=precision,ylabel='Held-out synthetic accuracy',ylim=(0,1))
            ax.legend(frameon=False)
        fig.suptitle('Acquisition toy: same label budget; bars = means, whiskers = seed range',fontsize=13,y=1.01)
        fig.tight_layout(); name='active_acquisition.png'; fig.savefig(directory/name,dpi=170,bbox_inches='tight'); plt.close(fig); figures.append(name)
    return figures
