"""Read retained selective-replay evidence, verify hashes, and render a report."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def requested_score(stage, request):
    prompt = request["prompt"].format(request["subject"])
    row = next(p for p in stage["probes"] if p["prompt"] == prompt)
    return row["targets"][request["target_new"]["str"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--prior-incomplete", type=Path)
    args = parser.parse_args()
    folder = args.run
    run = json.loads((folder / "run.json").read_text())
    data = json.loads((folder / "results.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("run incomplete")
    root = Path(__file__).resolve().parents[1]
    errors = []
    for name, digest in summary["binary_artifacts"].items():
        if sha(folder / name) != digest:
            errors.append("artifact: " + name)
    for name, digest in [("scripts/selective_replay.py", run["script_sha256"]), ("scripts/factual_edit_chain.py", run["receipt_helper_sha256"])]:
        blob = subprocess.check_output(["git", "show", run["source_commit"] + ":" + name], cwd=root)
        if hashlib.sha256(blob).hexdigest() != digest:
            errors.append("committed source: " + name)
    for path, mapping in [(args.model_dir, run["model_files"]), (args.upstream, run["upstream_files"])]:
        for name, digest in mapping.items():
            if sha(path / name) != digest:
                errors.append("pinned dependency: " + name)
    repeated_checks = 0
    if args.prior_incomplete:
        prior = json.loads((args.prior_incomplete / "results.json").read_text())
        def check_equal(old, new, label):
            nonlocal repeated_checks
            repeated_checks += 1
            if old != new:
                errors.append("storage-only rerun changed measurement: " + label)
        check_equal(prior["baseline"], data["baseline"], "baseline")
        lookup = {(r["algorithm"], r["pair"]["id"]): r for r in data["rows"]}
        for old in prior["rows"]:
            key = (old["algorithm"], old["pair"]["id"])
            new = lookup[key]
            for field in ["after_A", "after_AB", "reference_B_only", "B_update_relative_dependency"]:
                check_equal(old[field], new[field], str(key) + "/" + field)
            arms = {a["method"]: a for a in new["arms"]}
            for old_arm in old["arms"]:
                for field in ["result", "comparison_to_B_only"]:
                    check_equal(old_arm[field], arms[old_arm["method"]][field], str(key) + "/" + old_arm["method"] + "/" + field)
    audit = {"source_commit": run["source_commit"], "analysis_script_sha256": sha(__file__),
             "results_sha256": sha(folder / "results.json"), "binaries_checked": len(summary["binary_artifacts"]),
             "model_files_checked": len(run["model_files"]), "upstream_files_checked": len(run["upstream_files"]),
             "prior_incomplete_measurements_compared": repeated_checks, "errors": errors}
    (folder / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if errors:
        raise ValueError(errors)
    rows = data["rows"]
    methods = [r["method"] for r in rows[0]["arms"]]
    stats = {}
    for method in methods:
        arms = [a for row in rows for a in row["arms"] if a["method"] == method]
        comparisons = [a["comparison_to_B_only"] for a in arms]
        stats[method] = {"chains": len(arms), "state_bitwise": sum(c["state_bitwise"] for c in comparisons),
                         "logits_bitwise": sum(c["logits_bitwise"] for c in comparisons),
                         "all_generations_equal": sum(c["same_generations"] == c["probe_count"] for c in comparisons),
                         "mean_kl_range": [min(c["mean_kl_reference_to_arm"] for c in comparisons), max(c["mean_kl_reference_to_arm"] for c in comparisons)]}
    (folder / "comparison.json").write_text(json.dumps(stats, indent=2) + "\n")
    text = ["# Aを撤回したあと、Bは『Aなしで学習したB』になっているか", "",
            "固定したGPT-2 FP32・著者実装ROME/MEMITによる6系列の開発用診断。",
            "単純な可逆性や選択的撤回そのものの新規性は主張しない。",
            "[先行研究との比較](../../docs/PRIOR_WORK_20261110.md)と[事前手順](../../docs/SELECTIVE_REPLAY.md)を参照。", "",
            "## 比較する状態", "",
            "A→Bの順に実際の内部目標・更新因子を最適化する。その後Aを撤回し、",
            "最初からAを与えずBだけを最適化した参照モデル `E(W0, B)` と比べる。", "",
            "- `subtract_A`: 最終状態からAの因子をそのまま差し引く。",
            "- `exact_rewind_stale_B`: B→Aの順に厳密復元してW0を確認し、Aの存在下で作ったBの更新を使い回す。",
            "- `exact_rewind_recompute_B`: 同じ厳密復元の後、Bの目標・因子を最適化し直す。",
            "- `checkpoint_recompute_B`: 各履歴のcheckpointでB→Aを復元し、Bを最適化し直す。", "",
            "## Bだけ編集した参照への一致", "",
            "| 方法 | 全モデル状態 | 全初期logit | 全12問の生成 | 平均KLの範囲（参照→各方法） |", "| --- | ---: | ---: | ---: | ---: |"]
    for method, s in stats.items():
        lo, hi = s["mean_kl_range"]
        text.append(f"| {method} | {s['state_bitwise']}/{s['chains']} | {s['logits_bitwise']}/{s['chains']} | {s['all_generations_equal']}/{s['chains']} | {lo:.6g}–{hi:.6g} |")
    text += ["", "同じ実行環境・固定された設定での一致。すべての完全巻き戻しは中間WAとW0の全状態ハッシュも確認した。",
             "実際のAの厳密復元記録をWABへ直接適用する誤用は全系列で拒否された。単純引き算はその制約を持たない診断用の算術処理。", "",
             "### 更新の依存と丸め誤差を分ける", "",
             "| 方式 / 組 | B更新の相対差（層ごとの最小–最大） | 厳密巻き戻し＋古いBの平均KL | 同じ生成 / 12 |", "| --- | ---: | ---: | ---: |"]
    for row in rows:
        dep = list(row["B_update_relative_dependency"].values())
        arm = next(a for a in row["arms"] if a["method"] == "exact_rewind_stale_B")
        c = arm["comparison_to_B_only"]
        text.append(f"| {row['algorithm']} / {row['pair']['id']} | {min(dep):.6g}–{max(dep):.6g} | {c['mean_kl_reference_to_arm']:.6g} | {c['same_generations']}/{c['probe_count']} |")
    text += ["", "相対差は `||ΔB(WA)−ΔB(W0)|| / ||ΔB(W0)||`。これは重みの復元誤差とは別の量。",
             "厳密にW0へ戻した後も古いBを使うと差が残るため、この差を逆演算の丸め誤差だけでは説明できない。",
             "一方、後続編集の再計算は既存の再学習の原理を使う。今回の実験は依存の切り分けであり、新しいアルゴリズムの優越性の証明ではない。", "",
             "## 編集が実際に目標を出したか", "",
             "単語境界付きの先頭一致。続く文章の品質は保証しない。参照への完全一致は、編集自体の成功とは別。", "",
             "| 方式 / 組 | A直後のA目標 | A→B直後のB目標 | Bのみ参照のB目標 | 古いB再利用のB目標 | 新しいB再計算のB目標 |", "| --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        pair = row["pair"]
        stale = next(a["result"] for a in row["arms"] if a["method"] == "exact_rewind_stale_B")
        fresh = next(a["result"] for a in row["arms"] if a["method"] == "exact_rewind_recompute_B")
        scores = [requested_score(row["after_A"], pair["A"]), requested_score(row["after_AB"], pair["B"]), requested_score(row["reference_B_only"], pair["B"]), requested_score(stale, pair["B"]), requested_score(fresh, pair["B"])]
        text.append(f"| {row['algorithm']} / {pair['id']} | " + " | ".join(f"{'一致' if s['word_prefix'] else '不一致'} (p={s['probability']:.4g})" for s in scores) + " |")
    text += ["", "Aだけ撤回しても、保持したBがAの問いに影響することはあり得る。",
             "そのため、Aの答えを元に戻すことと、Aを含まない編集履歴を再現することは同じ目的ではない。",
             "全生成、元からの失敗、言い換え、近傍と4対照の確率は [results.json](results.json) に保存。", "",
             "## 復元記録の容量", "",
             "| 方式 / 組 | 2履歴checkpoint bytes | 直接XOR＋zlib bytes | 因子＋厳密補正 bytes | 単一W0 checkpoint bytes |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        sizes = {k: row["record_bytes_A"][k] + row["record_bytes_B"][k] for k in row["record_bytes_A"]}
        text.append(f"| {row['algorithm']} / {row['pair']['id']} | {sizes['checkpoint']} | {sizes['direct_xor_zlib']} | {sizes['delta_exact']} | {row['record_bytes_A']['checkpoint']} |")
    text += ["", "対象層の復元記録のみ。各コンテナと照合情報を含むが、共通モデル・共分散・リクエスト・測定用の重みコピーは含まない。",
             "2履歴を個別に撤回できる記録の比較であり、Aだけの撤回に特化すれば単一W0 checkpointへ直接戻す方が簡潔。最後の列はその保存容量の追加診断。",
             "測定用に保存した密なB更新・全重み・logitは別途ローカルに残っている。総ディスクやRAMがこの容量になるとは主張しない。",
             "直接XORの復元も検証した。この方式は同じzlibを使う基本対照で、ZipNNやULPackの実装・性能を再現したものではない。", "",
             "## 時間（単発測定）", "",
             "| 方式 / 組 | 厳密復元の壁時計秒 | B再最適化秒 | checkpoint復元の壁時計秒 | B再最適化秒 |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        arms = {a["method"]: a for a in row["arms"]}
        values = []
        for method in ["exact_rewind_recompute_B", "checkpoint_recompute_B"]:
            a = arms[method]
            values.extend([sum(t["parent_wall_seconds"] for t in a["undo_times"]), a["fresh_B_optimization_seconds"]])
        text.append(f"| {row['algorithm']} / {row['pair']['id']} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    text += ["", "復元時間はプロセス起動・ファイル入出力・層への反映を含む。最適化時間は因子生成、評価生成は含まない。",
             "各セル1回なので速度優位性や分散は主張しない。checkpoint比較も2履歴の逐次復元であり、直接W0を読む最速実装ではない。", "",
             "## 証拠と次の課題", "",
             f"事前コミット `{run['source_commit']}`。監査はバイナリ {audit['binaries_checked']}件、モデル {audit['model_files_checked']}件、著者コード {audit['upstream_files_checked']}件、実行ソースを確認し、エラー {len(errors)}件。",
             f"容量不足で中断した初回実行も残し、保存済みの数値・状態・生成について {repeated_checks}組を再実行結果と照合した。時間は再測定なので一致の対象外。",
             "[run.json](run.json)、[summary.json](summary.json)、[audit.json](audit.json)、全最適化ログを公開。大きい配列はignored `work/` に保存しハッシュを公開。", "",
             "この結果から先へ進む候補は、編集履歴のどこまで再最適化が必要かを検証できる形で限定し、そのための保存量と再計算量を測ること。",
             "長い履歴・強いモデル・独立パネル・既存手法の同条件実装比較は未実施。新規性は研究仮説として扱い、今回確立したとはしない。", ""]
    (folder / "RESULTS.md").write_text("\n".join(text))
    print(json.dumps({"audit": audit, "comparisons": stats}, indent=2))


if __name__ == "__main__":
    main()
