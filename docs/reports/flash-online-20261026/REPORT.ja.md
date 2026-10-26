# Kunlun K200 FlashAttention Online Softmax 最適化評価

対象: KunlunK200 0.7.0 → 0.7.1。実機: 192.168.50.112、KL1/K200、PD0/PD1、XRE 4.33、XTDK 2.7.98.2。日付: 2026-10-26 JST。

## 結果

Online SoftmaxのFP32分母の加算順序を維持したまま、PPのF16 mask変換とmax reductionを短縮した。代表PPの単体は **3.965 → 1.418 ms、2.80倍**。D128/S2048/M128/H32のF16 mask付きFlash APIは約19 → 8.6 ms。PP2048の実推論はTinyLlamaで約24%、Llama 2 7Bで約9.5〜9.7%改善した。

採用経路は検証した全形状の0.7.0比較とLLM logitsでビット一致した。SIMD並列sumは単体では速かったが、TinyLlama logitsに追加差が生じたため製品dispatchから除外した。SDNN KQ/KQV、KV writer、Q4、RoPE、SwiGLUの既存50個のkernel binaryは両buildともSHA256一致である。

## 1. 従来実装のボトルネック

生成ISAはKL1の`svmul_s`/`vvadd_s`が8 lane、比較・`exp_s`がscalar。KL1用公開SIMD headerにはpacked F32 expやhorizontal max/sumがなく、後継XPU用命令は使用していない。既存のSFU競合実験はcluster内共有、cluster間独立を示す。TG H32/M1は旧worker番号では2 clusterのみに偏っていた。

従来コードはF16 maskを要素ごとに汎用変換し、同じquery/broadcast maskを全headで再読・再変換していた。逐次maxには長い依存鎖があり、最大値不変時も`exp(0)`とacc全要素の乗算を実行していた。sumもscalarだが、後述の品質理由により順序を維持した。

PD0、D128、block512、M128、H32、F16 causal mask、途中blockの診断:

| 段階 | 旧 ms/平均worker | 新 ms/平均worker |
| --- | --- | --- |
| 前PVの取込み・状態load | 0.057433 | 0.055556 |
| Score L3 load | 0.022868 | 0.020956 |
| mask load・F16 decode | 2.375637 | 0.037972 |
| ALiBi・Q係数 | 0.003085 | 0.003820 |
| scale・mask SIMD | 0.121470 | 0.121450 |
| max reduction | 0.754826 | 0.524748 |
| alpha・acc rescale | 0.013246 | 0.023839 |
| native exp＋逐次sum | 0.464534 | 0.558248 |
| P/acc/stats store | 0.052408 | 0.048626 |


これは別コンパイルした`get_clock()`診断の **全行tick合計/64worker/900000**。最遅workerのwall timeではない。clock・診断出力・worker間の位相差を含むため、通常API時間との加減算には使わない。mask部分は約2.38 → 0.038 ms、maxは約0.75 → 0.52 ms。exp＋sumはこの診断ではむしろ増加しており、全段階が速くなったとは主張しない。

製品ループはexpとsumが同一ループである。別の`split`診断では8要素のexp/store後にsumを計測した（stage 7/10）。追加LM再読とclockを伴う補助実験であり、製品のexp/sum時間として置き換えない。最大値を100に固定するphase2診断も保存し、再スケール省略と通常の最大値更新を区別した。詳細は[stages.csv](stages.csv)。

## 2. 候補の比較と採否

| 候補 | 実測・精度 | 判断 |
| --- | --- | --- |
| F16 maskのpacked decode、不要な乗算省略、TG cluster分散 | 代表API 19.0 → 14.5 ms、旧出力ビット一致 | 採用候補の基礎 |
| 4本max＋8 lane並列sum | 約12.9 ms、単体旧比NMSE約4×10⁻¹³ | sumは不採用 |
| 上記＋head間mask再利用 | 約8.2 ms、単体は高精度 | 並列sum版は実験用のみ |
| 4本max＋head間mask再利用＋**従来順のFP32 sum** | 約8.55 ms、LLM/corpusともビット一致 | **製品採用** |
| 5次多項式exp・後継XPU専用SIMD exp | 前回通常Softmaxでは多項式が遅く、KL1用packed expも未確認 | 再採用せず。native expを維持 |

並列sum版はTinyLlamaのprefix128/512/2048でlogits NMSE約0.00096〜0.00098、平均KL約0.0052〜0.0061となった。単体の微小な丸め差だけでLLM品質を判断できない例である。量子化の閾値変化による増幅などは仮説であり、層別原因の特定は行っていない。この版の7B測定は品質判断で中止した。実験ソースは`bench/experiments/flash-online/`、過去ログは結果ディレクトリに残したが、製品build/install/dispatchには含めない。

## 3. 採用カーネルと契約

- `kunlun-flash-online.xpu`は0.7.0のまま保存。新しい`kunlun-flash-online-ordered.xpu`と`flash-online-device.h`を追加。
- `KUNLUN_FLASH_ONLINE=auto`または未設定が新経路、`legacy`が旧経路、`ordered`は新経路の明示alias。不明値はlegacyへ戻る。
- maxは4本の独立比較鎖に分け、最後に結合。分母sumは旧版と同じe0→e7の逐次FP32加算。native exp、確率の定義、FP32状態とaccを維持。
- 有限なrunning maxが変わらなければalpha=1とし、acc乗算を省略。NaN/+Infについて旧処理を崩さない条件分岐。末尾の8要素単位の−Infはexpと+0加算を省略。
- workerを`core_id*cluster_num+cluster_id`に変更。Mが64workerの倍数なら各workerがqueryを保持してheadを巡回し、同一offsetのbroadcast maskをLMで再利用する。その他は通常の行分割。
- mask cacheはkernel呼出しごとに無効化し、offset変更で再読する。ALiBi有効時は毎行再読し、既にslopeを乗じた値を再利用しない。KV更新や次blockには持ち越さない。
- F16入力をF32 mask workspaceの前半へ読み、32bit単位で後ろから展開。0/−Infのpacked pairを高速処理し、他は従来の半精度変換を使う。上書き前の入力を壊さず独立half配列を削除。
- 前PVをaccへ加算してからrunning maxに合わせてrescaleし、最終callでも最後のPVを取り込む。全mask blockでもこの取込みを省略しない。

SDNNとCLの共有L3ポインタ、score/P alias、partial/acc、32byte/行のstats、config ABI、default block（PP512/TG1024）・query tile128・GQA正規化方式は不変。kernel launch/wait追加なし。非対応shapeの既存backend fallbackも不変。

## 4. 単体・Flash API性能とメモリ

単体はD128/H32、PP block512、TG block1024、phase1。入力を毎回同じ状態へ戻し、その転送は計時外。20回の中央値、1回のlaunch＋completion waitを含む。小さいTGではhost waitの数十µsのばらつきがあり、最小値も併記する。

| PD | M | mask | 旧 ms | 新 ms | 速度比 | 最小値 旧→新 ms |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 128 | なし | 1.536602 | 1.246591 | 1.23x | 1.521653 → 1.235725 |
| 0 | 128 | F16 | 3.964627 | 1.417810 | 2.80x | 3.951811 → 1.413503 |
| 0 | 128 | F32 | 1.593366 | 1.380999 | 1.15x | 1.585718 → 1.377135 |
| 1 | 128 | F16 | 3.959611 | 1.414818 | 2.80x | 3.950092 → 1.411921 |
| 0 | 1 | F16 | 0.162643 | 0.104583 | 1.56x | 0.155172 → 0.080475 |
| 1 | 1 | F16 | 0.157377 | 0.080705 | 1.95x | 0.154967 → 0.079918 |


通常単体のenqueueは約2〜3 µs。単体kernelは1 launch/1 wait、Flash APIは各blockでKQ→online→PV、最後にonline finalを1回、warm時の完了waitは1回のまま。0.7.0の通常APIと同じ起動構造であり、前回の追加wait付き「online約1.62 ms」と直接比較しない。

Flash APIはcast/range/findmax、workspace準備、SDNN、online、最終waitまでを含む。各版warmup1回＋4回測定の上側中央値。代表PD0:

| D | S | M | mask | 旧 API ms | 新 API ms | 速度比 |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 512 | 128 | F16 | 4.631851 | 2.092882 | 2.21x |
| 64 | 2048 | 128 | F16 | 18.028363 | 7.803694 | 2.31x |
| 128 | 512 | 128 | F16 | 4.888204 | 2.356645 | 2.07x |
| 128 | 2048 | 128 | F16 | 19.008152 | 8.540139 | 2.23x |
| 128 | 4096 | 128 | F16 | 37.089410 | 17.006189 | 2.18x |
| 128 | 2048 | 128 | F32 | 9.455553 | 8.419336 | 1.12x |
| 128 | 2048 | 1 | F16 | 1.211709 | 1.106427 | 1.10x |
| 128 | 4096 | 1 | F16 | 2.393374 | 2.190332 | 1.09x |


全PD/形状の[units.csv](units.csv)を保存した。API差は単体だけでなくSDNN前処理や入力strideも含み、Online Softmax時間だけに読み替えない。

メモリ量は構造からの算定（バスカウンタ実測ではない）。D128/M128/H32/block512のL3 workspaceは両版12,713,984 byte＝12.125 MiB。Score読8 MiB＋P書8 MiB、acc/partial読4 MiB＋acc書2 MiB、stats読書0.25 MiB＝途中blockあたり22.25 MiBで不変。Score/PのL3アクセスを削る全面融合は行っていない。

同条件のhead broadcast F16 maskのGM読は4 MiB → 0.125 MiB（32分の1）、F32は8 MiB → 0.25 MiB。maskがHBMにある場合のHBM読削減である。ALiBi・head別mask・非64query倍数では再利用量が変わる。新たな中間HBM bufferやmetadata転送は追加しない。

コンパイル後frameは **11,584 → 9,568 byte**（compiler上限16,256 byte、LM/stack共用）。kernel binaryは3,276 → 4,476 byte、実際のcode allocationの増分は1,280 byte/loaded program。L3/workspace削減とcode size増加を混同しない。既存50 kernelのSHA一致は[final-verification.json](final-verification.json)に保存した。

## 5. 実LLMのPP/TG

TinyLlama 1.1B / Llama 2 7B、どちらも既存Q4_0 GGUF。Single PD、CPU threads6、context4096、batch/ubatch128、prefixは同じ保存token列。warmup1＋2反復、反復別token/sの中央値。TGは各prefix後のteacher-forced 15 stepの総時間から計算し、prefix長で表記する。PD0はlegacy→新、PD1は新→legacyの順。

`GGML_KUNLUN_ATTENTION=1`、`KUNLUN_ATTENTION_MATMUL=auto`、`KUNLUN_ATTENTION_SOFTMAX=fast`、`KUNLUN_ATTENTION_PREP=auto`、`KUNLUN_FLASH_KV=1`、`GGML_KUNLUN_FLASH=1`、`K200_FLASH=1`を共通にし、block/qt/PV/GQAの既定値を固定。SwiGLU設定0なども両版で共通。診断TRACE、CHECK_STATE、preload auditは通常性能測定に使用しない。

| Model | PD | prefix | PP 旧→新 token/s | PP改善 | TG 旧→新 token/s | TG改善 |
| --- | --- | --- | --- | --- | --- | --- |
| 7b | 0 | 128 | 68.98 → 70.65 | +2.41% | 7.097 → 7.127 | +0.43% |
| 7b | 0 | 512 | 67.59 → 70.13 | +3.76% | 6.006 → 6.020 | +0.23% |
| 7b | 0 | 2048 | 59.87 → 65.53 | +9.47% | 4.136 → 4.183 | +1.13% |
| 7b | 1 | 128 | 69.12 → 70.70 | +2.27% | 7.097 → 7.128 | +0.43% |
| 7b | 1 | 512 | 67.84 → 70.13 | +3.38% | 6.024 → 6.100 | +1.26% |
| 7b | 1 | 2048 | 60.04 → 65.86 | +9.70% | 4.150 → 4.218 | +1.62% |
| tiny | 0 | 128 | 265.72 → 282.36 | +6.26% | 22.000 → 21.937 | -0.29% |
| tiny | 0 | 512 | 255.07 → 279.30 | +9.50% | 19.965 → 20.233 | +1.34% |
| tiny | 0 | 2048 | 196.14 → 244.19 | +24.49% | 15.376 → 15.988 | +3.98% |
| tiny | 1 | 128 | 264.39 → 283.12 | +7.09% | 22.009 → 22.043 | +0.16% |
| tiny | 1 | 512 | 254.56 → 277.98 | +9.20% | 19.779 → 20.212 | +2.19% |
| tiny | 1 | 2048 | 196.15 → 243.88 | +24.33% | 15.216 → 16.006 | +5.19% |


TG@128の−0.29〜＋0.43%は小さく、2反復で統計的有意差を主張しない。単体の改善率ほどLLM全体は伸びず、特に7BはQ4/FFN・KQ/PV・前処理・launchが残る。各反復と範囲は[performance.csv](performance.csv)、[trials.csv](trials.csv)、改善率は[performance-comparison.csv](performance-comparison.csv)。

## 6. 数値精度・生成・互換性

両PD、D64/128、S256/512/2048/4096、M1/128、F16/F32 maskの64代表試験と、GQA・outer stride・interleaved KV・M3/S544端数・head/batch broadcast・ALiBi・zero/tiny/outlier・scale0/負・KV same-pointer更新など38試験、計102件で旧出力と全要素ビット一致。入力不変性とguardも通過。CPU FP64比較ではPPは4query/全head、TG/小形状は全行を参照し、FP16へ丸めたQを基準とする。最大NMSE 2.16974e-08、最大絶対誤差 3.51313e-05。これは既存SDNN量子化等を含み、新旧差はNMSE=0/最大絶対誤差=0。

追加4件は元のFP32 QをCPU FP64参照とし、既存cast丸めも誤差に含めた。最大NMSE 5.74004e-08、最大絶対誤差 2.30144e-05、旧版との差は引き続き0。Online単体ではNaN/±Inf/±10000、全mask、全65,536個のhalf bit patternを両PDで試し、確率・acc・statsのビット列がlegacyと一致した。これはCL単体の比較であり、SDNN API全体の非有限Q/K/V対応を新たに保証するものではない。

1024 token corpus（1020位置）では新旧の全logitsがビット一致:

| Model | 0.7.0 / 0.7.1 perplexity | 新旧 NMSE / KL / max abs |
| --- | --- | --- |
| TinyLlama | 11.1246713081 / 11.1246713081 | 全て0 |
| Llama 2 7B | 8.34594649553 / 8.34594649553 | 全て0 |

CPU Attention比のcorpus NMSE/KLはTiny約0.00235/0.01105、7B約0.00614/0.00957で0.7.0と同じ。既存量子化・演算経路の差を含み、INT16単独の誤差とは断定しない。今回の品質差の増分は0。[ordered-corpus-quality.json](ordered-corpus-quality.json)に比較値とSHAを保存。

通常性能試験のprefix128/512/2048後のlogitsも全て一致。Single/Layer/Row Split、両モデル、prefix128/512の32位置、短文32 token・長prefix2048後64 tokenのgreedy生成で一致し、全state restore診断の次logits一致を確認。KV更新・再利用と既存SDNNの互換性を確認した。非FA stateとの相互変換は対象外。Attention/RoPE/SwiGLUの既存実機回帰を両PDで通過し、Flash/Attention/Q4 layout contractも通過。[compat-final.json](compat-final.json)。

## 7. 運用・保存と再現

作業前は両repoがclean。branchは`codex/k200-flash-online-20261026`、baselineはlibrary `780fb6933fe3d4ee06e686438f5f37fc7826d901`、llama.cpp `736fad58aa9209b38c5aced9ce11948ed7a0561f`。before tar/patch・git状態と0.7.0 buildを保存し、新buildを別ディレクトリに作成した。今回コミット/pushは行っていない。

実機結果: `/home/unagidojyou/kunlun-k200/results/flash-online-20261026`。製品source: `repos/KunlunK200`、llama.cppの追加変更は利用文書のみ。新build: `build/flash-online-lib` / `build/flash-online-llama`。既存`flash-lib` / `flash-llama`は保持。コンパイル時のdevice関数指定漏れを修正したが、デバイスエラーやハングはなし。品質差による探索ジョブ中止は記録済み。リセット0回、最終両PD RUNNING/no error。

再現は[REPRODUCE.ja.md](REPRODUCE.ja.md)を参照。ソース、CL assembly、通常/診断ログ、CSV、scriptと入力token列を成果物へ保存した。モデル・SDK・private SSH鍵は配布物へ含めない。

## 8. 残存ボトルネックと次段階

1. native expはscalar/SFU共有、sumは品質維持のため逐次FP32。単体での高速性だけで並列sumを採用せず、層別logits/量子化の閾値を追跡してから精度方式を再検討する。
2. maxの比較・LM loadは依然scalar。KL1の実命令で独立比較鎖やload schedulingを詰める余地がある。現在の4本より多いaccumulatorは未評価で、最適本数の証明はしていない。
3. Score/P・accのL3 load/store、blockごとの3 launch、SDNN前処理は残る。次段階ではKQ改善やSDNN/CL pipelineを別課題として評価する。今回のbuffer寿命・順序契約は維持した。
4. TGではOnline単体の改善がLLMへ反映されにくく固定launch/Q4/長context KQが支配的。Flash-Decodingや複数coreのKV block分割は今回実装していない。
5. LLM検証はprefix2048＋64生成まで、単体S4096まで。より長context、大規模batch、多様なモデルやcorpusでの品質保証・統計的な性能比較は今後の課題。
