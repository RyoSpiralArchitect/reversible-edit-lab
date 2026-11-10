"""Audit saved component/replay evidence and render bounded study results."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def latest_scores(history, observation):
    latest = {r['prompt'].format(r['subject']): r['target_new']['str'] for r in history}
    lookup = {p['prompt']: p for p in observation['probes']}
    return {p: {'target': t, **lookup[p]['targets'][t]} for p, t in latest.items()}


def controls(observation):
    rows = [p for p in observation['probes'] if p['id'].startswith('control_')]
    return {r['id']: next(iter(r['targets'].values()))['word_prefix'] for r in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--prior-incomplete', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = args.run
    manifest = json.loads((folder / 'run.json').read_text())
    data = json.loads((folder / 'results.json').read_text())
    artifacts = json.loads((folder / 'artifacts.json').read_text())
    if data['status'] != 'complete' or len(data['decomposition']) != 6 or len(data['histories']) != 12:
        raise ValueError('incomplete fixed panel')
    errors = []
    repeated_groups = 0
    if args.prior_incomplete:
        prior = json.loads((args.prior_incomplete / 'results.json').read_text())
        lookup = {(r['algorithm'], r['pair']['id']): r for r in data['decomposition']}
        def same(old, new, label):
            nonlocal repeated_groups
            repeated_groups += 1
            if old != new:
                errors.append('rerun measurement changed: ' + label)
        same(prior['baseline'], data['baseline'], 'baseline')
        for old in prior['decomposition']:
            key = old['algorithm'], old['pair']['id']
            new = lookup[key]
            for field in ['reference', 'adaptation_comparison', 'component_differences']:
                same(old[field], new[field], str(key) + '/' + field)
            by_mode = {a['mode']: a for a in new['arms']}
            for arm in old['arms']:
                for field in ['result', 'comparison']:
                    same(arm[field], by_mode[arm['mode']][field], str(key) + '/' + arm['mode'] + '/' + field)
    for name, digest in artifacts.items():
        if sha(folder / name) != digest:
            errors.append('artifact: ' + name)
    for name, digest in manifest['sources'].items():
        blob = subprocess.check_output(['git', 'show', manifest['source_commit'] + ':' + name], cwd=root)
        if hashlib.sha256(blob).hexdigest() != digest:
            errors.append('source: ' + name)
    for path, mapping in [(args.model_dir, manifest['model_files']), (args.upstream, manifest['upstream_files'])]:
        for name, digest in mapping.items():
            if sha(path / name) != digest:
                errors.append('pinned dependency: ' + name)
    lookup_checks = 0
    for row in data['decomposition']:
        if not row['adaptation_comparison']['state_bitwise']:
            errors.append('adaptation did not match author')
        for trial in row['timing_repeats']:
            for mode in ['author', 'checked']:
                if not trial['arms'][mode]['state_bitwise']:
                    errors.append('timing arm state mismatch')
        infos = [a['info'] for a in row['arms'] if a['mode'] == 'checked']
        infos += [t['arms']['checked'] for t in row['timing_repeats']]
        for info in infos:
            for event in info['events']:
                lookup_checks += 1
                if event['hit'] and not event['dependency_equal']:
                    errors.append('invalid checked reuse')
    for row in data['histories']:
        checked = next(a for a in row['arms'] if a['mode'] == 'checked')
        for info in checked['infos']:
            for event in info['events']:
                lookup_checks += 1
                if event['hit'] and not event['dependency_equal']:
                    errors.append('invalid long-history reuse')
    audit = {'errors': errors, 'binaries_checked': len(artifacts), 'model_files_checked': len(manifest['model_files']),
             'upstream_files_checked': len(manifest['upstream_files']), 'source_files_checked': len(manifest['sources']),
             'checked_lookup_decisions': lookup_checks, 'prior_measurement_groups_compared': repeated_groups,
             'results_sha256': sha(folder / 'results.json'),
             'analysis_source_sha256': sha(__file__)}
    (folder / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    if errors:
        raise ValueError(errors)
    comparisons = {}
    for algorithm in ['rome', 'memit']:
        rows = [r for r in data['decomposition'] if r['algorithm'] == algorithm]
        comparisons[algorithm] = {}
        for mode in ['fresh', 'old_target', 'old_keys', 'old_both', 'old_update', 'checked']:
            values = [next(a['comparison'] for a in r['arms'] if a['mode'] == mode) for r in rows]
            comparisons[algorithm][mode] = {'cells': len(values), 'state_matches': sum(v['state_bitwise'] for v in values),
                'logit_matches': sum(v['logits_bitwise'] for v in values),
                'all_generation_matches': sum(v['same_generations'] == v['probe_count'] for v in values),
                'kl_range': [min(v['mean_kl'] for v in values), max(v['mean_kl'] for v in values)]}
    timings = []
    for row in data['decomposition']:
        pairs = row['timing_repeats']
        author = [p['arms']['author']['edit_seconds'] for p in pairs]
        checked = [p['arms']['checked']['edit_seconds'] for p in pairs]
        checks = [p['arms']['checked']['check_seconds'] for p in pairs]
        timings.append({'algorithm': row['algorithm'], 'pair': row['pair']['id'], 'repeats': len(pairs),
            'author_median_seconds': statistics.median(author), 'checked_median_seconds': statistics.median(checked),
            'check_median_seconds': statistics.median(checks),
            'paired_ratio_median': statistics.median(b/a for a, b in zip(author, checked)),
            'paired_ratios': [b/a for a, b in zip(author, checked)]})
    history_summary = []
    for row in data['histories']:
        retained = [r for i, r in enumerate(manifest['history'][:row['length']]) if i != row['deleted_index']]
        scored = latest_scores(retained, row['reference'])
        history_summary.append({'algorithm': row['algorithm'], 'length': row['length'], 'deleted_index': row['deleted_index'],
            'arms': {a['mode']: a['comparison'] for a in row['arms']},
            'latest_request_scores': scored, 'reference_controls': controls(row['reference']),
            'reference_target_successes': sum(s['word_prefix'] for s in scored.values()),
            'latest_request_count': len(scored),
            'checked_cache_hits': sum(i['cache_hits'] for a in row['arms'] if a['mode'] == 'checked' for i in a['infos']),
            'checked_cache_slots': sum(i['cache_slots'] for a in row['arms'] if a['mode'] == 'checked' for i in a['infos'])})
    summary = {'comparisons': comparisons, 'timings': timings, 'histories': history_summary,
               'baseline_controls': controls(data['baseline'])}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# 何を再計算すれば、過去の編集を正しく取り消せるか', '',
        '固定GPT-2 CPU FP32・著者ROME/MEMITの開発実験。6条件の分解、18組の時間比較、',
        '8件・16件の履歴から計12回の選択的撤回を実施。新規性・一般能力・SOTAの証明ではない。', '',
        '[事前手順](../../docs/DEPENDENCY_REPLAY.md) / [実行manifest](run.json) / [全結果](results.json) / [監査](audit.json)', '',
        '## 1. Aの影響がBのどの計算に残るか', '',
        'A→BからB、Aを厳密に巻き戻し、Bの内部目標とキー側計算を別々に再利用する。',
        'ROMEでは内部目標と最終値因子を区別し、現在出力の差し引き・正規化を毎回計算する。',
        'MEMITでは層ごとのキーと調整済みキーの線形求解を対象とし、現在状態に対する残差は毎回計算する。', '',
        '| 方法 | ROME 全状態一致 | MEMIT 全状態一致 |', '| --- | ---: | ---: |']
    labels = {'fresh': 'すべて再計算', 'old_target': '内部目標だけ再利用', 'old_keys': 'キー側だけ再利用',
              'old_both': '内部目標＋キー側を再利用', 'old_update': '更新全体を再利用', 'checked': '依存を確認して再利用'}
    for mode, label in labels.items():
        lines.append('| ' + label + ' | ' + ' | '.join(f"{comparisons[a][mode]['state_matches']}/3" for a in ['rome', 'memit']) + ' |')
    lines += ['', '比較対象は「Aを与えず最初からBだけ編集したモデル」。',
              '計測用アダプタの無再利用実行も、未改変の著者実装に全6条件で全状態一致した。', '',
              '| 方式 / 組 | 内部目標の相対差 | キー側のbit一致 |', '| --- | ---: | --- |']
    for row in data['decomposition']:
        diff = row['component_differences']
        target = next(v['relative_difference'] for k, v in diff.items() if k.startswith('target:'))
        keys = ', '.join(k + '=' + str(v['bitwise']) for k, v in diff.items() if not k.startswith('target:'))
        lines.append(f"| {row['algorithm']} / {row['pair']['id']} | {target:.6g} | {keys} |")
    lines += ['', '## 2. 再利用条件と、確認を含めた実時間', '',
        '内部目標は全モデルテンソル、キーは当該GPT-2層のc_proj入力までのテンソル、',
        '線形求解は実際のキー・共分散に依存する。設定・要求・コード・固定資源も結び付ける。',
        'live bytesのハッシュを毎回計算する保守的な実装。変更なし、上流変更、同層c_proj変更、',
        '要求変更、キャッシュ破損、MEMIT共分散変更の実モデル対照を実施した。', '',
        '| 方式 / 組 | 著者実装 秒 | 確認付き再利用 秒 | うち確認 秒 | 対応比の中央値 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for t in timings:
        lines.append(f"| {t['algorithm']} / {t['pair']} | {t['author_median_seconds']:.4f} | {t['checked_median_seconds']:.4f} | {t['check_median_seconds']:.4f} | {t['paired_ratio_median']:.3f}× |")
    lines += ['', '各条件3回、順序を交互にして実行。比は「確認付き／著者実装」で、1より大きいと遅い。',
        '時間には編集内の依存確認・値の検査・計算・重み適用を含む。共通の計測用全状態ハッシュ、',
        '生成評価、アダプタ構築、artifact I/Oは除外。初回キャッシュ生成費は別途results.jsonに保存する。',
        '単一端末・固定順序の小規模測定で、一般的な速度差や統計的優越性は主張しない。', '',
        '## 3. 長い履歴の途中を撤回', '',
        '元の履歴先端から撤回位置まで厳密復元し、保持する後続要求だけを再実行する。',
        '参照はW0から当該要求を除外してすべて新しく編集したモデル。', '',
        '| 方式 | 履歴長 | 撤回index | 新規suffix 状態/logit | 確認付き 状態/logit | 古い更新 状態/logit | 再利用 / 判定 | 参照の最終要求成功 |',
        '| --- | ---: | ---: | --- | --- | --- | ---: | ---: |']
    for r in history_summary:
        c, s, f = r['arms']['checked'], r['arms']['old_update'], r['arms']['fresh_suffix']
        lines.append(f"| {r['algorithm']} | {r['length']} | {r['deleted_index']} | {f['state_bitwise']}/{f['logits_bitwise']} | {c['state_bitwise']}/{c['logits_bitwise']} | {s['state_bitwise']}/{s['logits_bitwise']} | {r['checked_cache_hits']}/{r['checked_cache_slots']} | {r['reference_target_successes']}/{r['latest_request_count']} |")
    lines += ['', '要求成功は同じ問いへの最後の保持要求を対象とする。競合した古い要求を同時に正解とは数えない。',
        '参照との一致と編集自体の成功を分離する。基準制御問の正解状況はsummary.jsonに保存。',
        '後続B等が撤回対象の問いに影響することもあり、撤回した答えが出ないことだけを判定に使わない。',
        '長履歴では同じ巻き戻し位置から未再利用でsuffixだけを再計算する対照も測る。',
        '時間は単回でfresh_suffix先行。W0からの全面再計算との比較にはprefix省略の効果も混ざる。', '',
        '| 方式 / 長さ / 撤回 | 新規suffix 秒 | 確認付きsuffix 秒 | 新規巻き戻し 秒 | 確認付き巻き戻し 秒 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for row in data['histories']:
        arms = {a['mode']: a for a in row['arms']}
        f, c = arms['fresh_suffix'], arms['checked']
        lines.append(f"| {row['algorithm']} / {row['length']} / {row['deleted_index']} | {f['replay_seconds']:.4f} | {c['replay_seconds']:.4f} | {f['undo_seconds']:.4f} | {c['undo_seconds']:.4f} |")
    lines += ['',
        '## 保存費用と検証範囲', '', '| 方式 | 履歴長 | 正確な復元記録 bytes | 成分cache bytes |', '| --- | ---: | ---: | ---: |']
    seen = set()
    for row in data['histories']:
        key = row['algorithm'], row['length']
        if key not in seen:
            seen.add(key)
            lines.append(f"| {key[0]} | {key[1]} | {row['journal_bytes']} | {row['component_bytes']} |")
    lines += ['', '共通のモデル・tokenizer・共分散、計測用の重みコピー・logit、実行中のRAMは上表の外。',
        '復元記録には旧状態の丸め補正情報を保持する。情報なしの忘却・復元ではない。',
        '同一の保存bytesはhard linkで共有するが、数値計算は実行する。大きな配列はgit対象外でローカル保持。',
        '証明範囲は固定のeager GPT-2、eval、CPU、単一書き手、不変の非テンソル設定に限定する。',
        '別モデル、適応する共分散、永続optimizer、外部の書き換え、別backendの保証には拡張していない。', '',
        f"監査: {audit['binaries_checked']} binary、{audit['model_files_checked']} model、{audit['upstream_files_checked']} upstream、{audit['source_files_checked']}実行元ファイル、{lookup_checks}件の再利用判定。エラー0。", '',
        f"初回中断は対照群の保存メタデータ修正によるもの。途中ログ・artifactは残した。再実行と{repeated_groups}組の既存測定を比較した（時間を除外）。"]
    (folder / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
