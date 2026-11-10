# 新しい対応づけを一時的に重みに書き、同じ問いを繰り返す

固定した GPT-2 FP32・著者実装 ROME/MEMIT による3件ずつの小規模試験。
その場で作った48-bitの架空プロジェクト名に2語の合言葉を割り当て、モデルを試す前にパネルと手順をコミットした。
合言葉を推論プロンプトに含めず、同じ穴埋め質問で8トークンまでgreedy生成した。
新規性は単語そのものではなく新しい任意の対応づけ。実際の認証情報ではない。

## 同じ質問への出力

質問: `The access phrase for Project <nonce> is`。以下は省略せず保存した生成文字列（改行は `\n`）。

| 方式 / ケース | 目標 | 編集前 | 編集中 | 厳密復元後 |
| --- | --- | --- | --- | --- |
| rome / novel-1 | ` velvet meadow` | ` "I'm not sure if this is` | ` velvet meadow meadow meadow me` | ` "I'm not sure if this is` |
| rome / novel-2 | ` frozen sparrow` | ` "I'm not sure what you mean` | ` frozen sparrows and cubensis lame` | ` "I'm not sure what you mean` |
| rome / novel-3 | ` amber forest` | ` "The Access Program is a program that` | ` amber forest forest forest forest forest forest forest` | ` "The Access Program is a program that` |
| memit / novel-1 | ` velvet meadow` | ` "I'm not sure if this is` | ` a quiet summer day in the summertime` | ` "I'm not sure if this is` |
| memit / novel-2 | ` frozen sparrow` | ` "I'm not sure what you mean` | ` frozen sparrows;\n\nJul` | ` "I'm not sure what you mean` |
| memit / novel-3 | ` amber forest` | ` "The Access Program is a program that` | ` amber forest forest forest forest forest forest forest` | ` "The Access Program is a program that` |

事前の判定は『目標のトークン列が生成の先頭に完全一致する』というもの。
ただし `frozen sparrows` は `frozen sparrow` のトークン列で始まるため、この判定では通る。
出力確認後に単語境界も要求する補助判定を追加した。元の判定を置き換えず併記し、再学習・ケースの差し替えはしていない。
どちらも短い先頭句の再現であり、続く反復や不自然な文章まで良質になったことは意味しない。

| 方式 | 事前のトークン先頭一致 | 単語境界も一致（事後診断） | 言い換えで単語境界まで一致 |
| --- | ---: | ---: | ---: |
| rome | 3/3 | 2/3 | 1/3 |
| memit | 2/3 | 1/3 | 2/3 |

## 復元の確認

| 復元方法 | 全モデル状態が元と一致 | 18問の全初期logitが一致 | 全18生成が元と一致 |
| --- | ---: | ---: | ---: |
| checkpoint | 6/6 | 6/6 | 6/6 |
| delta | 0/6 | 0/6 | 6/6 |
| delta_exact | 6/6 | 6/6 | 6/6 |

全方式で復元後の同じ問いは目標の先頭一致を示さなかった。しかし通常の差分引き算は元の重み・logitに一致しなかった。
**『答えが出なくなった』だけでは完全復元の証拠として足りない。** 厳密補正とcheckpointは保存ビットまで戻り、生成も元に戻った。
復元workerが受け取るのは編集後行列と各方式の記録のみ。厳密補正記録には編集前状態の情報が含まれる。
次の編集は厳密復元されたモデルから直接始めた。元モデルの再ロード・隠れた重みのリセットは行っていない。

## 他の問いへの影響

12問の固定対照のうち元モデルが目標句を出したのは4問。元から誤っている問いも残した。
『他の知能がすべて正常』を示す能力試験ではない。編集中の回答変化と、復元後の元状態への一致を分けて測った。

| 方式 / ケース | 元の正答4問の保持 | 対照12問の生成が不変 | 対照の平均KL（base→edited） | 別プロジェクトへの合言葉漏出 / 2問 |
| --- | ---: | ---: | ---: | ---: |
| rome / novel-1 | 4/4 | 7/12 | 0.016778 | 0/2 |
| rome / novel-2 | 4/4 | 5/12 | 0.00534804 | 0/2 |
| rome / novel-3 | 4/4 | 8/12 | 0.00663156 | 0/2 |
| memit / novel-1 | 4/4 | 8/12 | 0.0138506 | 0/2 |
| memit / novel-2 | 4/4 | 7/12 | 0.00885321 | 0/2 |
| memit / novel-3 | 4/4 | 6/12 | 0.00590168 | 2/2 |

同種の別プロジェクトまで編集対象の合言葉を返すなら、対応づけの局所性が崩れている。
厳密復元後は対照12問も含め全18問が元と同一。短いパネルの正答保持と、あらゆる能力の保存を同一視しない。

### 元モデルの対照回答

| 問い | 目標句 | 元の生成 | 目標先頭一致 |
| --- | --- | --- | --- |
| `2 + 2 =` | ` 4` | ` 3.5 + 3.5 +` | False |
| `5 + 3 =` | ` 8` | ` 1.5 + 3 = 1.` | False |
| `1, 2, 3, 4,` | ` 5` | ` 5, 6, 7, 8,` | True |
| `Monday, Tuesday, Wednesday,` | ` Thursday` | ` Thursday, Friday, Saturday, Sunday,` | True |
| `The opposite of hot is` | ` cold` | ` cold.\n\nThe coldest part` | True |
| `The opposite of up is` | ` down` | ` down.\n\nThe first thing to` | True |
| `Roses are red, violets are` | ` blue` | ` green, and the color of the leaves` | False |
| `The quick brown fox jumps over the lazy` | ` dog` | `, lazy fox and they both fall to` | False |
| `The capital of France is` | ` Paris` | ` the capital of the French Republic, and` | False |
| `The capital of Japan is` | ` Tokyo` | ` the capital of the United States, and` | False |
| `The capital of Germany is` | ` Berlin` | ` the capital of the German state, and` | False |
| `The largest planet in the solar system is` | ` Jupiter` | ` about 1.5 billion light-years` | False |

## 記録容量

各対象層のNPZ記録の合計。checkpoint・差分因子・厳密補正それぞれのメタデータと圧縮コンテナを含む。モデル全体の容量ではない。

| 方式 / ケース | checkpoint bytes | delta bytes | delta_exact bytes | exact / checkpoint |
| --- | ---: | ---: | ---: | ---: |
| rome / novel-1 | 8770096 | 15069 | 255928 | 2.92% |
| rome / novel-2 | 8770094 | 15097 | 263805 | 3.01% |
| rome / novel-3 | 8770089 | 15066 | 254185 | 2.90% |
| memit / novel-1 | 26308220 | 83727 | 526804 | 2.00% |
| memit / novel-2 | 26308213 | 83694 | 550254 | 2.09% |
| memit / novel-3 | 26308217 | 83656 | 549941 | 2.09% |

## 証拠と範囲

- 事前コミット: `05d446ead8cafb789622ab291cda035647d7a71b`。3件×2方式、CPU FP32。
- [固定手順](../../docs/NOVEL_INFORMATION.md)、[パネル](../../experiments/novel_information_panel.json)、[run.json](run.json)、[全生成・スコア](results.json)、[追加診断・監査](analysis.json)。
- バイナリ 149件、実行コード・パネルのコミット一致を確認。 モデルファイル: 6件を確認。 著者コード: 21件を確認。 実施した監査のエラー: 0件。
- 最適化ログを保存。重み・因子・補正記録・全初期logitはローカルのignored `work/` に保存し、[summary.json](summary.json)にSHA-256を公開。
- 合言葉の生成成功、言い換え、局所性、復元を別々に評価。小さい共分散校正とGPT-2用設定であり、原論文のベンチマーク再現や汎用知能の検証ではない。
