# Reversible Edit Lab v0.2

[![tests](https://github.com/RyoSpiralArchitect/reversible-edit-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/RyoSpiralArchitect/reversible-edit-lab/actions/workflows/tests.yml)

ROME/MEMITの局所書き込みを、**編集 → 復元 → 次の学習**まで調べる研究実装。

v0.2では、通常の逆演算で残る丸め誤差を圧縮して記録し、**元の行列全体を復元時に渡さず、保存ビットまで戻す**機能を追加しました。必要な補助情報、実際のファイル容量、反復によるずれ、次の問い合わせ先への影響を一緒に測ります。

基本パッケージは合成した線形連想記憶と分類器を扱い、NumPyで動きます。追加の任意スクリプトでは、GPT-2の実勾配と著者実装のROME/MEMITによる目標最適化・多層更新も試しています。これらにはTorch・Transformers等の追加環境が必要です。実モデルの結果と限界は末尾の各試験を参照してください。

## すぐに動かす

Python 3.10以上。NumPyがある環境ではインストールせず実行できます。

```bash
python3 run_experiments.py refine --out results/my_v02 --plots
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

新しく環境を作る場合：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[plots]"
python run_experiments.py refine --out results/my_v02 --plots
```

短い確認には次を使えます。`--plots`を省けばmatplotlibは不要です。

```bash
python3 run_experiments.py refine --out results/quick --seeds 7 --precisions bfloat16 --cycles 16 --active-rounds 2
```

出力先は新しいディレクトリを指定してください。既存の実験結果を上書きしません。

## v0.2で加えたもの

| 機能 | 内容 |
| --- | --- |
| 厳密な丸め補正 | 通常の復元と元状態の保存ビットのXORをzlibで可逆圧縮 |
| 予算付き部分補正 | 数値誤差の大きい要素から保存。補正データ予算を5/10/25%で比較 |
| 実容量の比較 | 因子・補正・ハッシュ・NPZヘッダーを含む記録と、圧縮checkpointを比較 |
| 512回の編集・撤回 | 直前の生きた状態から毎回記録し、撤回後にその記録を破棄 |
| 能動学習への接続 | 問い合わせ系列、候補スコア、最終重みをcheckpoint方式と比較 |
| 保存済み記録の連続復元 | hard ROMEの8段の編集を逆順に戻し、途中状態のハッシュも検証 |

**補正記録には旧状態の情報が含まれます。** 記録作成時には編集前状態を使い、復元時には編集後状態と宣言された記録だけを使います。過去に無記録で編集されたモデルから、未知の元行列を取り出す機能ではありません。

## 復元方法と入力

| CLIの方式 | 復元に渡すもの | 条件 |
| --- | --- | --- |
| `delta` | 編集後行列と低ランク差分因子 | 有限精度では差し引き後に誤差が残る |
| `delta_exact` | 上記と圧縮したビット補正 | hard ROMEにも対応。予測復元の再現が必要 |
| `delta_sparse` | 差分因子と一部要素のビット補正 | 誤差と記録予算を交換。完全復元は一般に保証しない |
| `analytic` | 編集後行列とキー・目標・書き込み条件 | ROMEは`alpha < 1`。MEMITは固定目標の単一行列 |
| `analytic_exact` | 上記と圧縮したビット補正 | 解析逆変換の数値誤差も補正する |
| `old_response` | ROME条件と編集前の局所応答 | hard ROMEの失われた応答を明示的に保存 |
| `svd` | 編集後行列と事前に固定した除去ランク | 重みだけを使うヒューリスティック |

`checkpoint`は評価基準として用意しています。小さな行列や、密な共分散を毎回保存するMEMIT解析方式では、復元記録がcheckpointより大きくなる場合があります。

## Python API：保存ビットまで戻す

```python
import numpy as np
from reversible_edit.core import quantize, rome_edit
from reversible_edit.receipts import make_receipt, restore_from_receipt, tensor_digest

rng = np.random.default_rng(7)
W = quantize(rng.normal(size=(64, 128)), "bfloat16")
k = rng.normal(size=128)
v = rng.normal(size=64)
C = np.eye(128)
expected_digest = tensor_digest(W)  # 評価用のダイジェストだけ残す

edit = rome_edit(W, k, v, C, alpha=1.0, precision="bfloat16")
receipt = make_receipt(
    W, edit, editor="rome", recovery="delta_exact", precision="bfloat16"
)
edited = edit.updated
del W, edit, k, v, C

restored = restore_from_receipt(edited, receipt)
assert tensor_digest(restored) == expected_digest
```

行列は`(output_dim, input_dim)`、バッチキーは`(input_dim, edit_count)`。関数は入力配列を変更しません。

演算と因子・共分散はfloat64で、重みの書き込みを指定精度に丸めます。bfloat16は直接の最近接偶数丸めをエミュレートし、float32配列に格納します。容量計測ではBF16の論理16ビット表現を使います。GPUのBF16積和演算や速度の再現ではありません。

厳密補正には、最初から宣言した精度で保存された状態が必要です。`quantize`で初期化した配列を使います。例えば未丸めのfloat64入力をBF16と宣言した場合は、黙って元状態を変えずエラーにします。

## ファイル単位で編集・復元する

ROME入力NPZには`weight`, `key`, `value`, `covariance`を入れます。MEMITでは`keys`と`values`を使います。

```bash
python3 run_experiments.py edit --input request.npz --output edited.npz --receipt receipt.npz --editor rome --recovery delta_exact --alpha 1 --precision bfloat16
python3 run_experiments.py undo --edited edited.npz --receipt receipt.npz --output restored.npz
```

2段の編集をファイルで保存し、元の入力を削除してから逆順に復元する例：

```bash
python3 examples/exact_chain.py --out demo_exact
```

`undo`は編集後の行列と記録だけを読みます。記録は編集後状態・予測復元・復元先のダイジェストと対応付け、誤った順序や改変を検出します。NPZは`allow_pickle=False`で読み、出力は新規作成します。

厳密補正による連続復元では、記録作成時と同じ保存ビットの予測復元を再計算する必要があります。演算環境や因子が変わって予測が異なる場合は拒否します。通常の`delta`や`analytic`では丸め誤差により先行記録のハッシュが合わなくなることがあります。

逆順の巻き戻しと、途中の一件だけを消して後続学習を残す操作は別です。後続のキー・目標・更新が過去の状態に依存する場合、後続の更新を再計算します。

## ベンチマークと結果

[v0.2の結果レポート](results/20261110_v02/RESULTS.md)に、3 seeds、3保存精度、2行列サイズでの比較を収録しています。

- 単発比較846件。厳密補正は234/234件で元の保存ビットと一致。
- 512回の編集・撤回では、FP32・FP16・BF16の両厳密補正方式が全ステップで一致。
- 64×128のROME、alpha=.95では、`delta_exact`の全記録は圧縮checkpointの20.6%（FP32）、38.3%（FP16）、44.4%（BF16）。3seedの平均。
- 6回の能動学習では全方式で問い合わせ系列が基準と一致。厳密補正方式は最終重みも一致し、近似復元方式は一致しなかった。
- hard ROMEの8段の保存済み記録を、最後の行列から逆順にビット一致で復元。

能動学習の取得方針はv0.1と同じ、未ラベルprobeの期待エントロピー減少です。本物のラベルは問い合わせ先を選んだ後に取得します。変更したのは撤回の忠実さであり、取得方針自体の優位性は別の比較事項です。

元のv0.1の結果は[こちら](results/20261110/RESULTS.md)。当時のソースと結果は`provenance/v0.1.0-source-and-results.zip`にそのまま保存しました。v0.1の実験構成は、現在のコードでも`run`サブコマンドで実行できます。

```bash
python3 run_experiments.py run --out results/legacy_design --plots
```

## 容量と精度の読み方

- `tradeoffs.csv`の`receipt_npz_bytes`は全記録の実ファイルサイズ。比較先の`checkpoint_npz_bytes`も圧縮済みです。
- 部分補正の5/10/25%は、元行列の論理バイト数に対する**補正データのみの予算**。因子・メタデータ・ヘッダーを加えた総量は別に計測します。
- active実験の容量は基底配列・スカラーと補正パケットの合計です。基底記録のNPZコンテナを含まないため、単発実験の全NPZ比率と混同しません。
- 反復の総生成バイト数は、破棄済みの記録も足した量です。同時に保持する記録容量とは異なります。
- 評価器は元行列や一時コピーを保持します。ファイルが小さいことからプロセスのピークメモリ削減や高速化を結論しません。
- 相対重み誤差はFrobeniusノルムの比です。「復元した重みの割合」ではありません。ビット一致は別の指標です。
- 単発比較の`alpha=0`は無更新の対照です。記録容量の主表には`alpha=.95`の書き込みを使います。

## ファイル構成

| ファイル | 内容 |
| --- | --- |
| `src/reversible_edit/core.py` | 更新・復元の数値カーネル、低精度丸め |
| `src/reversible_edit/roundoff.py` | 保存ビットの可逆圧縮補正、部分補正、整合確認 |
| `src/reversible_edit/receipts.py` | 必要な情報だけを保存するNPZ編集記録 |
| `src/reversible_edit/roundoff_experiments.py` | v0.2の容量・精度・反復・学習比較 |
| `src/reversible_edit/roundoff_report.py` | 生データからの集計・レポート・図 |
| `src/reversible_edit/active.py` | 仮更新と撤回による問い合わせ実験 |
| `src/reversible_edit/experiments.py` | v0.1の実験構成 |
| `tests/` | 数学的性質、丸め、情報境界、連続復元の検証 |
| `docs/ROUNDOFF.md` | v0.2の補正式、情報契約、容量の定義 |
| `docs/METHODS.md` | 固定ターゲット逆変換の数式と適用範囲 |
| `docs/REAL_MODEL.md` | 実LLMへの接続と検証事項 |
| `references.bib` | 原論文と関連する一次資料 |

実モデルの行列・更新因子、記録作成時間・ピークメモリの追加比較を以下に記録しています。問い合わせ方針やTTTの学習信号の改善は、可逆性とは独立した未検証の課題です。

## 追加実験：時間とプロセスメモリ

`costs` は checkpoint・差分記録・厳密補正付き差分記録を、方式・反復ごとに
独立したプロセスで比較します。記録の構築・圧縮、復元、プロセス全体のピークRSSを計測します。
復元側には編集後行列と記録だけを渡します。既存の行列カーネルの動作は変更していません。

```bash
.venv/bin/python -I -m reversible_edit.cli costs --out results/my_costs
```

標準設定は3サイズ、3精度、3seed、3反復の243比較です。生成用の合成重みと記録は
出力先の `work/` に残し、Git には計測値とハッシュを保存します。
計測範囲と事前に固定した条件は [docs/COSTS.md](docs/COSTS.md)、
実モデルへの段階的な接続は [docs/ROADMAP.md](docs/ROADMAP.md) を参照してください。

[最初の追加実験](results/20261110_costs/INTERPRETATION.md)は243比較を完了しました。
厳密補正は81/81件でビット一致。1024×2048では圧縮checkpointに対して記録容量は
8.7〜18.1%でしたが、記録作成時間は9.1〜20.1倍、復元時のプロセスピークRSSは
1.8〜2.9倍でした。合成行列・単一ホストでの観測で、実LLMの速度やメモリ削減を示すものではありません。

ライセンスは [Apache-2.0](LICENSE) です。元の配布物と検証記録の対応は
[provenance/README.md](provenance/README.md) を参照してください。

## GPT-2 の実際の勾配による追加比較

[GPT-2 試験](results/20261110_gpt2/RESULTS.md)では、最終MLP射影の重みと
実際のforward/backwardから得たランク1更新因子で81比較を実行しました。
厳密補正は27/27回で重みのビットが一致し、9条件すべてで復元後のlogitも一致しました。
更新量は重みノルムの0.1%に固定しています。これは単一トークン損失の勾配更新で、
ROME/MEMITの内部目標最適化や、事実編集の汎化性能を検証したものではありません。
条件・再現手順は [docs/GPT2_PILOT.md](docs/GPT2_PILOT.md) に記録しています。

## 著者実装による事実編集 → 復元 → 次の編集

[ROME/MEMITの追加試験](results/20261110_factual_chain/RESULTS.md)では、固定した著者実装を
CPU・GPT-2向けに移植し、内部目標最適化とMEMITの3層への残差配分を実行しました。
両方式とも、厳密補正で戻した全モデル状態と、そこから再計算した次の編集の状態・logitが
checkpoint基準に一致しました。通常の差分復元では一致しませんでした。

一方、この小規模な共分散推定・層設定では近傍の国の回答まで変わり、局所性は保てませんでした。
可逆性の確認を編集品質の成功とは扱いません。条件は [docs/FACTUAL_CHAIN.md](docs/FACTUAL_CHAIN.md)、
最適化ログ・全指標・負の結果は上記レポートに保存しています。

## 新規情報の一時書き込み → 同じ問い → 復元 → 同じ問い

[新規対応づけの試験](results/20261110_novel_information/RESULTS.md)では、その場で作った
架空のプロジェクト名3件に合言葉を割り当て、答えを推論プロンプトに含めず生成させました。
単語境界まで合う合言葉の先頭生成はROMEで2/3件、MEMITで1/3件。
事前のトークン先頭一致では3/3・2/3でしたが、`sparrow` と `sparrows` を区別する
事後診断では各1件を成功に数えていません。続く語の反復やMEMITの別名への漏出も残しています。

厳密復元は6/6件で全モデル状態・18問のlogitと生成を元に戻しました。
通常の差分引き算でも合言葉は出なくなり、生成は元と一致しましたが、重み・logitは不一致。
**答えが消えたことだけでは完全復元の判定になりません。** 元から正答できた対照4問は
編集中も全件保持しましたが、対照全12問の生成には変化があり、一般的な能力保存の証明ではありません。
固定した手順は [docs/NOVEL_INFORMATION.md](docs/NOVEL_INFORMATION.md) に記録しています。
