"""Report the fixed run, retaining a stricter post-run word-boundary diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def phrase_prefix(generated, target):
    """Reject a word extension such as 'frozen sparrows' for 'frozen sparrow'."""
    return bool(target) and re.match(re.escape(target) + r"(?=$|[\s.,!?;:\"')\]])", generated) is not None


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def cell(value):
    return "`" + str(value).replace("\n", "\\n").replace("|", "\\|").replace("`", "'") + "`"


def audit_description(audit):
    parts = [f"バイナリ {audit['binary_files_checked']}件、実行コード・パネルのコミット一致を確認。"]
    for label, key in [("モデルファイル", "model_files_checked"), ("著者コード", "upstream_files_checked")]:
        count = audit[key]
        parts.append(f"{label}: {count}件を確認。" if count else f"{label}: 未実施（パス指定なし）。")
    parts.append(f"実施した監査のエラー: {len(audit['errors'])}件。")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--upstream", type=Path)
    args = parser.parse_args()
    folder = args.run
    run = json.loads((folder / "run.json").read_text())
    data = json.loads((folder / "results.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    base = {p["id"]: p for p in data["baseline"]["probes"]}
    errors = []
    for name, digest in summary["binary_artifacts"].items():
        if sha(folder / name) != digest:
            errors.append("binary mismatch: " + name)
    root = Path(__file__).resolve().parents[1]
    for name, digest in [("scripts/novel_information.py", run["script_sha256"]),
                         ("scripts/factual_edit_chain.py", run["receipt_helper_sha256"]),
                         ("experiments/novel_information_panel.json", run["panel_sha256"])]:
        source = subprocess.check_output(["git", "show", run["source_commit"] + ":" + name], cwd=root)
        if hashlib.sha256(source).hexdigest() != digest:
            errors.append("committed source mismatch: " + name)
    if args.model_dir:
        for name, digest in run["model_files"].items():
            if sha(args.model_dir / name) != digest:
                errors.append("model mismatch: " + name)
    if args.upstream:
        for name, digest in run["upstream_files"].items():
            if sha(args.upstream / name) != digest:
                errors.append("upstream mismatch: " + name)
    rows = []
    for row in data["rows"]:
        key = row["case_id"] + "/rewrite"
        edited = next(p for p in row["edited"]["probes"] if p["id"] == key)
        paraphrase = next(p for p in row["edited"]["probes"] if p["id"] == row["case_id"] + "/paraphrase")
        exact = next(r for r in row["restorations"] if r["method"] == "delta_exact")
        restored = next(p for p in exact["restored"]["probes"] if p["id"] == key)
        leaks = [p["id"] for p in row["edited"]["probes"] if p["kind"] == "rewrite" and p["id"] != key and phrase_prefix(p["generated"], edited["target"])]
        rows.append({"algorithm": row["algorithm"], "case_id": row["case_id"],
                     "target": edited["target"], "before": base[key]["generated"],
                     "during": edited["generated"], "after_exact": restored["generated"],
                     "predeclared_token_prefix": edited["target_prefix_match"],
                     "posthoc_phrase_boundary": phrase_prefix(edited["generated"], edited["target"]),
                     "paraphrase_token_prefix": paraphrase["target_prefix_match"],
                     "paraphrase_phrase_boundary": phrase_prefix(paraphrase["generated"], edited["target"]),
                     "other_project_questions_returning_current_phrase": leaks,
                     "exact_assessment": exact["assessment"],
                     "edited_controls": row["edited_controls"],
                     "control_first_token_kl_mean": sum(p["first_token_kl_base_to_current"] for p in row["edited"]["probes"] if p["kind"] == "control") / row["edited_controls"]["total"],
                     "record_bytes": {r["method"]: sum(c["record_bytes"] for c in r["costs"]) for r in row["restorations"]}})
    derived = {"analysis_script_sha256": sha(__file__),
               "run_results_sha256": sha(folder / "results.json"),
               "boundary_diagnostic": "Added after inspecting fixed-run output; accepts complete target at start followed by end, whitespace or listed punctuation. No tuning or reruns. Rejects frozen sparrows for frozen sparrow; prefix metric retained.",
               "rows": rows, "audit": {"source_commit": run["source_commit"],
               "binary_files_checked": len(summary["binary_artifacts"]),
               "model_files_checked": len(run["model_files"]) if args.model_dir else 0,
               "upstream_files_checked": len(run["upstream_files"]) if args.upstream else 0,
               "errors": errors}}
    (folder / "analysis.json").write_text(json.dumps(derived, indent=2, allow_nan=False) + "\n")
    if errors:
        raise ValueError(errors)
    text = ["# 新しい対応づけを一時的に重みに書き、同じ問いを繰り返す", "",
            "固定した GPT-2 FP32・著者実装 ROME/MEMIT による3件ずつの小規模試験。",
            "その場で作った48-bitの架空プロジェクト名に2語の合言葉を割り当て、モデルを試す前にパネルと手順をコミットした。",
            "合言葉を推論プロンプトに含めず、同じ穴埋め質問で8トークンまでgreedy生成した。",
            "新規性は単語そのものではなく新しい任意の対応づけ。実際の認証情報ではない。", "",
            "## 同じ質問への出力", "",
            "質問: `The access phrase for Project <nonce> is`。以下は省略せず保存した生成文字列（改行は `\\n`）。", "",
            "| 方式 / ケース | 目標 | 編集前 | 編集中 | 厳密復元後 |", "| --- | --- | --- | --- | --- |"]
    for row in rows:
        text.append("| " + " | ".join([row["algorithm"] + " / " + row["case_id"]] + [cell(row[k]) for k in ["target", "before", "during", "after_exact"]]) + " |")
    text += ["", "事前の判定は『目標のトークン列が生成の先頭に完全一致する』というもの。",
             "ただし `frozen sparrows` は `frozen sparrow` のトークン列で始まるため、この判定では通る。",
             "出力確認後に単語境界も要求する補助判定を追加した。元の判定を置き換えず併記し、再学習・ケースの差し替えはしていない。",
             "どちらも短い先頭句の再現であり、続く反復や不自然な文章まで良質になったことは意味しない。", "",
             "| 方式 | 事前のトークン先頭一致 | 単語境界も一致（事後診断） | 言い換えで単語境界まで一致 |", "| --- | ---: | ---: | ---: |"]
    for algorithm in ["rome", "memit"]:
        selected = [r for r in rows if r["algorithm"] == algorithm]
        text.append(f"| {algorithm} | {sum(r['predeclared_token_prefix'] for r in selected)}/{len(selected)} | {sum(r['posthoc_phrase_boundary'] for r in selected)}/{len(selected)} | {sum(r['paraphrase_phrase_boundary'] for r in selected)}/{len(selected)} |")
    text += ["", "## 復元の確認", "",
             "| 復元方法 | 全モデル状態が元と一致 | 18問の全初期logitが一致 | 全18生成が元と一致 |", "| --- | ---: | ---: | ---: |"]
    for method in ["checkpoint", "delta", "delta_exact"]:
        selected = [r["assessment"] for row in data["rows"] for r in row["restorations"] if r["method"] == method]
        text.append(f"| {method} | " + " | ".join(f"{sum(r[k] for r in selected)}/{len(selected)}" for k in ["full_state_restored", "probe_logits_restored", "all_generations_restored"]) + " |")
    text += ["", "全方式で復元後の同じ問いは目標の先頭一致を示さなかった。しかし通常の差分引き算は元の重み・logitに一致しなかった。",
             "**『答えが出なくなった』だけでは完全復元の証拠として足りない。** 厳密補正とcheckpointは保存ビットまで戻り、生成も元に戻った。",
             "復元workerが受け取るのは編集後行列と各方式の記録のみ。厳密補正記録には編集前状態の情報が含まれる。",
             "次の編集は厳密復元されたモデルから直接始めた。元モデルの再ロード・隠れた重みのリセットは行っていない。", "",
             "## 他の問いへの影響", "",
             "12問の固定対照のうち元モデルが目標句を出したのは4問。元から誤っている問いも残した。",
             "『他の知能がすべて正常』を示す能力試験ではない。編集中の回答変化と、復元後の元状態への一致を分けて測った。", "",
             "| 方式 / ケース | 元の正答4問の保持 | 対照12問の生成が不変 | 対照の平均KL（base→edited） | 別プロジェクトへの合言葉漏出 / 2問 |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        controls = row["edited_controls"]
        text.append(f"| {row['algorithm']} / {row['case_id']} | {controls['baseline_matches_retained']}/{controls['baseline_target_matches']} | {controls['identical_generations']}/{controls['total']} | {row['control_first_token_kl_mean']:.6g} | {len(row['other_project_questions_returning_current_phrase'])}/2 |")
    text += ["", "同種の別プロジェクトまで編集対象の合言葉を返すなら、対応づけの局所性が崩れている。",
             "厳密復元後は対照12問も含め全18問が元と同一。短いパネルの正答保持と、あらゆる能力の保存を同一視しない。", "",
             "### 元モデルの対照回答", "", "| 問い | 目標句 | 元の生成 | 目標先頭一致 |", "| --- | --- | --- | --- |"]
    for probe in data["baseline"]["probes"]:
        if probe["kind"] == "control":
            text.append("| " + " | ".join([cell(probe[k]) for k in ["prompt", "target", "generated"]] + [str(probe["target_prefix_match"])]) + " |")
    text += ["", "## 記録容量", "", "各対象層のNPZ記録の合計。checkpoint・差分因子・厳密補正それぞれのメタデータと圧縮コンテナを含む。モデル全体の容量ではない。", "",
             "| 方式 / ケース | checkpoint bytes | delta bytes | delta_exact bytes | exact / checkpoint |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        sizes = row["record_bytes"]
        text.append(f"| {row['algorithm']} / {row['case_id']} | {sizes['checkpoint']} | {sizes['delta']} | {sizes['delta_exact']} | {sizes['delta_exact']/sizes['checkpoint']:.2%} |")
    text += ["", "## 証拠と範囲", "",
             f"- 事前コミット: `{run['source_commit']}`。3件×2方式、CPU FP32。",
             "- [固定手順](../../docs/NOVEL_INFORMATION.md)、[パネル](../../experiments/novel_information_panel.json)、[run.json](run.json)、[全生成・スコア](results.json)、[追加診断・監査](analysis.json)。",
             "- " + audit_description(derived["audit"]),
             "- 最適化ログを保存。重み・因子・補正記録・全初期logitはローカルのignored `work/` に保存し、[summary.json](summary.json)にSHA-256を公開。",
             "- 合言葉の生成成功、言い換え、局所性、復元を別々に評価。小さい共分散校正とGPT-2用設定であり、原論文のベンチマーク再現や汎用知能の検証ではない。", ""]
    (folder / "RESULTS.md").write_text("\n".join(text))
    print(json.dumps(derived["audit"], indent=2))


if __name__ == "__main__":
    main()
