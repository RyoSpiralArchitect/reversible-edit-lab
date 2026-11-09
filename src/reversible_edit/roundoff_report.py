"""Human-readable v0.2 evidence with physical record-size comparisons."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import numpy as np


LABELS={"delta":"Delta log", "delta_exact":"Delta + exact correction",
        "analytic":"Analytic inverse", "analytic_exact":"Analytic + exact correction",
        "delta_sparse_0.05":"Sparse 5% payload", "delta_sparse_0.1":"Sparse 10% payload",
        "delta_sparse_0.25":"Sparse 25% payload", "checkpoint":"Checkpoint",
        "factor":"Delta log", "factor_exact":"Delta + exact correction", "factor_sparse":"Sparse 25% payload"}
COLORS={"delta":"#00838a", "delta_exact":"#25633e", "analytic":"#aa59a8", "analytic_exact":"#3155a0",
        "delta_sparse_0.05":"#d8ab59","delta_sparse_0.1":"#cc8c31","delta_sparse_0.25":"#ba6d14"}


def _mean(rows,key): return float(np.mean([r[key] for r in rows]))


def summarize_refinement(data):
    cycle_groups=defaultdict(list)
    for row in data['cycle_rows']:
        if row['cycle']==data['config']['cycle_count']:
            cycle_groups[(row['precision'],row['method'])].append(row)
    cycles=[]
    for (precision,method),rows in cycle_groups.items():
        cycles.append({'precision':precision,'method':method,'n_seeds':len(rows),
                       'mean_relative_error':_mean(rows,'weight_relative_error'),
                       'bitwise_start_matches':sum(r['bitwise_restored_to_start'] for r in rows),
                       'exact_single_steps':sum(r['exact_single_steps'] for r in rows),
                       'total_single_steps':sum(r['cycle'] for r in rows),
                       'next_update_bitwise_matches':sum(r['next_update_bitwise_equal'] for r in rows),
                       'mean_receipt_npz_bytes':_mean(rows,'mean_receipt_npz_bytes')})
    cost_groups=defaultdict(list)
    for row in data['tradeoff_rows']:
        if (row['editor']=='rome' and row['alpha']==.95) or (row['editor']=='memit' and row['covariance_scale']==.1):
            cost_groups[(row['editor'],row['output_dim'],row['input_dim'],row['precision'],row['method'])].append(row)
    costs=[]
    for (editor,m,n,precision,method),rows in cost_groups.items():
        costs.append({'editor':editor,'output_dim':m,'input_dim':n,'precision':precision,'method':method,
                      'n_seeds':len(rows),'mean_receipt_npz_bytes':_mean(rows,'receipt_npz_bytes'),
                      'mean_checkpoint_npz_bytes':_mean(rows,'checkpoint_npz_bytes'),
                      'mean_ratio':_mean(rows,'receipt_to_compressed_checkpoint'),
                      'mean_relative_error':_mean(rows,'weight_relative_error'),
                      'bitwise_matches':sum(r['bitwise_restored'] for r in rows)})
    active_groups=defaultdict(list)
    for run in data['active_runs']:
        for path in run['active_paths']:
            if path['candidate_order']=='forward':
                active_groups[(run['config']['precision'],path['protocol'])].append(path)
    active=[]
    for (precision,method),rows in active_groups.items():
        active.append({'precision':precision,'method':method,'n_seeds':len(rows),
                       'mean_final_accuracy':_mean(rows,'final_test_accuracy'),
                       'query_sequences_match':sum(r.get('selected_sequence_matches_checkpoint',False) for r in rows),
                       'final_state_matches':sum(r.get('final_state_bitwise_matches_checkpoint',False) for r in rows)})
    # Flags are also recomputed from public data; never infer equality from accuracy.
    for entry in active:
        query_matches=state_matches=0
        for run in data['active_runs']:
            if run['config']['precision']!=entry['precision']: continue
            baseline=next(p for p in run['active_paths'] if p['protocol']=='checkpoint' and p['candidate_order']=='forward')
            candidate=next(p for p in run['active_paths'] if p['protocol']==entry['method'] and p['candidate_order']=='forward')
            query_matches+=candidate['selected_pool_ids']==baseline['selected_pool_ids']
            state_matches+=candidate['final_state_sha256']==baseline['final_state_sha256']
        entry['query_sequences_match']=query_matches
        entry['final_state_matches']=state_matches
    exact_rows=[r for r in data['tradeoff_rows'] if r['method'] in ('delta_exact','analytic_exact')]
    return {'cycle_count':data['config']['cycle_count'],'cycles':cycles,'costs':costs,'active':active,
            'tradeoff_comparisons':len(data['tradeoff_rows']),'cycle_observations':len(data['cycle_rows']),
            'exact_single_write_count':len(exact_rows),
            'exact_single_write_successes':sum(r['bitwise_restored'] for r in exact_rows),
            'receipt_chains':data['receipt_chains']}


def write_refinement_report(data,directory:Path,plots:bool=False):
    summary=summarize_refinement(data)
    (directory/'summary.json').write_text(json.dumps(summary,allow_nan=False,indent=2),encoding='utf-8')
    lines=['# Reversible Edit Lab v0.2 — 丸め補正の再実験','',
           '## 問いと実装','',
           '通常の復元の後に残る丸め誤差を、編集時に作る補助記録で補正した。補正の作成時には編集前状態を使う。復元時には元行列を渡さず、編集後行列と復元記録のみを使う。これは元情報を一切使わない盲目的な復元ではない。','',
           '厳密補正は、元行列と予測復元の保存ビットのXORをzlibで可逆圧縮する。部分補正は数値誤差の大きい要素を選ぶ。部分補正の5/10/25%は元行列の論理保存バイト数に対する補正データ予算であり、メタデータや因子を含む総量ではない。以下の容量比較にはすべてを含む実際のNPZサイズを使う。','',
           f"設定: seeds={data['config']['seeds']}, precisions={data['config']['precisions']}, shapes={data['config']['shapes']}。",
           f"単発比較 {summary['tradeoff_comparisons']}件、反復観測 {summary['cycle_observations']}件。厳密補正の単発ビット一致は {summary['exact_single_write_successes']}/{summary['exact_single_write_count']}。",'',
           '## 反復して戻した後','',
           f"ROME alpha=.95、編集→撤回を{summary['cycle_count']}回。反復中に使った記録は撤回直後に破棄する。重み誤差はseed間の平均。",'',
           '| 保存精度 | 方式 | 相対重み誤差 | 全ステップの厳密復元 | 次の同一更新までビット一致 |',
           '| --- | --- | ---: | ---: | ---: |']
    for r in summary['cycles']:
        lines.append(f"| {r['precision']} | {LABELS.get(r['method'],r['method'])} | {r['mean_relative_error']:.6g} | {r['exact_single_steps']}/{r['total_single_steps']} | {r['next_update_bitwise_matches']}/{r['n_seeds']} |")
    lines+=['','## 役立つ強さの書き込みと記録容量','',
            'ROMEはalpha=.95、MEMITは保持共分散を0.1倍にした条件。各値は3seed等の設定seed数で平均。チェックポイントもNPZ圧縮済みで、BF16は16ビットの論理表現を保存する。メタデータの再利用・共分散の償却は仮定しない。alpha=0の無更新対照はこの表に含めない。','',
            '| 編集器 | 行列 | 精度 | 方式 | 復元記録 bytes | 圧縮checkpoint bytes | 比率 | 相対重み誤差 |',
            '| --- | --- | --- | --- | ---: | ---: | ---: | ---: |']
    for r in summary['costs']:
        if r['method'] not in ('delta','delta_exact','delta_sparse_0.25','analytic_exact'): continue
        lines.append(f"| {r['editor']} | {r['output_dim']}×{r['input_dim']} | {r['precision']} | {LABELS.get(r['method'],r['method'])} | {r['mean_receipt_npz_bytes']:.1f} | {r['mean_checkpoint_npz_bytes']:.1f} | {100*r['mean_ratio']:.1f}% | {r['mean_relative_error']:.4g} |")
    lines+=['','小さい行列ではヘッダーやダイジェストが支配し、補正記録がチェックポイントを上回り得る。MEMITの解析方式は密な共分散も毎回含む。容量の優位は一律ではなく、サイズ・更新強度・保存精度・記録方式の条件を持つ。','']
    if summary['active']:
        lines+=['## 次に何を問い合わせるか','',
                'v0.1と同じ期待エントロピー減少の取得方針で、復元の変更が判断と最終状態へどう伝わるかを測定した。教師ラベルは選択後だけ取得。取得方針の性能優位を新たに主張する実験ではない。表は順方向評価、逆順も生データに含む。','',
                '| 精度 | 方式 | 最終分類精度 | 問い合わせ系列がcheckpointと一致 | 最終重みもビット一致 |',
                '| --- | --- | ---: | ---: | ---: |']
        for r in summary['active']:
            lines.append(f"| {r['precision']} | {LABELS.get(r['method'],r['method'])} | {100*r['mean_final_accuracy']:.2f}% | {r['query_sequences_match']}/{r['n_seeds']} | {r['final_state_matches']}/{r['n_seeds']} |")
    lines+=['','## 連続する保存済み記録の巻き戻し','',
            'hard ROMEで8回連続して編集し、最後の行列とシリアライズ済み記録だけを逆順の復元処理へ渡した。途中の全状態ダイジェストまで比較する。','']
    for r in summary['receipt_chains']:
        lines.append(f"- {r['precision']}: {r['steps']}段、途中状態すべて一致={r['every_intermediate_digest_matches']}、初期状態とビット一致={r['final_bitwise_restored']}。")
    lines+=['','厳密補正は、作成時と同じビット列の予測復元を再計算することが条件。異なる演算環境等で予測が変わればダイジェスト検査で拒否する。後続編集を残して途中の編集だけを差し引く機能とは別で、意味の依存関係は再適用時に扱う必要がある。','',
            '## 結果の範囲','',
            '- 合成した固定ターゲットの線形更新。事前学習済みLLMの実行は行っていない。',
            '- ビット一致は補正の設計から期待される性質。実験で確かめた価値は、その実装と記録容量、反復、後続学習の整合。',
            '- 補助記録には旧状態の情報を保持する。無記録の重み復元を達成したものではない。',
            '- 圧縮ファイル容量とプロセスのピークメモリは別。評価器は元行列や一時コピーを保持する。',
            '- 処理時間には補正構築・ハッシュ・圧縮・復元の費用がある。記録が小さいことから高速化を結論しない。','',
            '## 証拠ファイル','',
            '`results.json`に設定・全生データ・ソースハッシュ、`tradeoffs.csv`に容量と誤差、`cycles.csv`に反復経過、`summary.json`に本ページの集計を収録。','']
    if plots:
        for filename in plot_refinement(data,summary,directory):
            lines.extend([f'![{filename}]({filename})',''])
    (directory/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    return summary


def plot_refinement(data,summary,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
                         'grid.alpha':.15,'figure.facecolor':'#fafaf8','axes.facecolor':'#fafaf8','savefig.facecolor':'#fafaf8'})
    figures=[]
    precisions=data['config']['precisions']
    fig,axes=plt.subplots(1,len(precisions),figsize=(4.3*len(precisions),4.1),squeeze=False)
    for ax,p in zip(axes[0],precisions):
        for method in ('delta','delta_sparse_0.25','analytic','delta_exact','analytic_exact'):
            selected=[r for r in data['cycle_rows'] if r['precision']==p and r['method']==method]
            steps=sorted({r['cycle'] for r in selected})
            vals=[max(_mean([r for r in selected if r['cycle']==s],'weight_relative_error'),1e-18) for s in steps]
            ax.loglog(steps,vals,marker='o',label=LABELS[method],color=COLORS[method],linestyle='--' if 'exact' in method else '-')
        ax.set(title=p,xlabel='Edit / undo cycles',ylabel='Relative weight error')
        ax.legend(frameon=False,fontsize=7,loc='center left')
    fig.suptitle('Roundoff correction removes accumulated drift (zero shown at 1e-18)',fontsize=13,y=1.02)
    fig.tight_layout(); name='corrected_cycles.png';fig.savefig(directory/name,dpi=160,bbox_inches='tight');plt.close(fig);figures.append(name)
    fig,axes=plt.subplots(1,2,figsize=(12.4,4.2))
    for ax,editor in zip(axes,('rome','memit')):
        combos=[(m,n,p) for m,n in data['config']['shapes'] for p in precisions]
        x=np.arange(len(combos))
        for offset,method in ((-.18,'delta_exact'),(.18,'analytic_exact')):
            vals=[]
            for m,n,p in combos:
                rs=[r for r in summary['costs'] if r['editor']==editor and r['method']==method
                    and r['output_dim']==m and r['input_dim']==n and r['precision']==p]
                vals.append(rs[0]['mean_ratio'] if rs else np.nan)
            ax.bar(x+offset,vals,width=.34,label=LABELS[method],color=COLORS[method],alpha=.9)
        ax.axhline(1,color='#4a4a4a',linestyle='--',label='Compressed checkpoint')
        ax.set_xticks(x,[f'{m}x{n}\n{p}' for m,n,p in combos],rotation=30,ha='right')
        ax.set(title='ROME alpha=.95' if editor=='rome' else 'MEMIT C scale=.1',ylabel='Complete NPZ record / NPZ checkpoint')
        ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Exact recovery has a size-dependent cost; all metadata is counted',fontsize=13,y=1.02)
    fig.tight_layout();name='exact_record_cost.png';fig.savefig(directory/name,dpi=170,bbox_inches='tight');plt.close(fig);figures.append(name)
    fig,axes=plt.subplots(1,len(precisions),figsize=(4.3*len(precisions),3.9),squeeze=False)
    max_m,max_n=max(data['config']['shapes'],key=lambda t:t[0]*t[1])
    for ax,p in zip(axes[0],precisions):
        for method in ('delta','delta_sparse_0.05','delta_sparse_0.1','delta_sparse_0.25','delta_exact'):
            rs=[r for r in summary['costs'] if r['editor']=='rome' and r['precision']==p and r['output_dim']==max_m
                and r['input_dim']==max_n and r['method']==method]
            if not rs:continue
            r=rs[0];x=r['mean_ratio'];y=r['mean_relative_error']
            ax.scatter([x],[y],s=65,color=COLORS[method],label=LABELS[method],zorder=3)
        ax.set(title=p,xlabel='Complete record / compressed checkpoint',ylabel='Relative weight error after undo',ylim=(0,None))
        ax.legend(frameon=False,fontsize=7)
    fig.suptitle(f'Record budget and fidelity: {max_m}x{max_n}, ROME alpha=.95',fontsize=13,y=1.02)
    fig.tight_layout();name='correction_tradeoff.png';fig.savefig(directory/name,dpi=170,bbox_inches='tight');plt.close(fig);figures.append(name)
    return figures
