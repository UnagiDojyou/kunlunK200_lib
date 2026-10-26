# Kunlun K200 FlashAttention KV最大値キャッシュ（0.7.4）

## 結果

FlashAttention 0.7.3の演算・量子化・Online Softmax・Flash-Decodingを維持し、K/V最大値の取得経路を変更した。初回はSDK findmaxで正確な最大値をseedし、以後は**有限性が確認済みのKVに対する、未使用のゼロ行への追加**だけをGPU上で増分更新する。上書き・不明な書き込み・非対応viewには従来経路を使用する。

性能はKV writer込みのLLM推論で判断した。単体の結果と推論結果、通常測定と追加waitを伴う診断値を分けて以下に示す。入力をFP16へ丸める順序、INT16 scale、MACの演算順序は変更していない。

Llama 2 7BのTG@4096は 3.766 → 4.125 token/s（+9.53%）。TinyLlamaはTG@8192でPD0 +1.74%、PD1 +5.63%を観測した。Tiny 4096のcache候補は全体の利益が出なかったため、自動有効化を8192以上へ引き上げた。全比較でlogitsはbit一致し、perplexity・KLの悪化は確認されなかった。

## 1. 比較条件と保護した成果

- 実機: `192.168.50.112`、K200 KL1、PD0/PD1、各PD HBM約8064 MiB・L3 16 MiB。
- SDK: XRE 4.33、XTDK 2.7.98.2、既存KL1 XDNN。温度・他プロセス確認は既存runner/thermal guardで毎試験実施。
- 比較元: ライブラリ `41bbc1242d7e4d12776de1692c15a0008198d2aa`、llama.cpp `90d64614e749240d271dbef2863d588157ceb825`。
- 旧build `build/flash-decode-lib` / `build/flash-decode-llama` を保持。変更前の差分・ソースsnapshotは `results/kv-max-20261026/*-before*` と `*-before.tar.gz` に保存した。
- 新build: `build/kv-max-lib` / `build/kv-max-llama`。作業branchは両repoとも `codex/k200-kv-max-20261026`。
- TinyLlama 1.1B / Llama 2 7B Q4_0。同一ID入力・teacher token・context・batchで比較。通常性能試験にはTRACE、verify、LD_PRELOAD profiler、追加waitを入れていない。

基本設定は `GGML_KUNLUN_ATTENTION=1`, `KUNLUN_ATTENTION_MATMUL=auto`, `KUNLUN_FLASH_KV=1`, `GGML_KUNLUN_FLASH=1`, `KUNLUN_FLASH_KQ=auto`, `KUNLUN_FLASH_ONLINE=auto`。Wide KQV、共通Q4、RoPE等の設定も揃えた。Singleを中心に、Layer/Row SplitとFlash-Decoding off/autoも比較した。

## 2. 旧findmaxの構成と費用

`src/flash.inc` の共通前処理で、pack/cast後のFP16 K/VそれぞれにSDK `findmax<float16>` を1回ずつ実行していた。KV blockごとではなくAttention APIごとに合計2回であり、Flash-Decodingもこの同じ前処理を使用していた。

SDKは4つのFP32部分最大値を出力する。既存KQ/KQVは4値をすべてreduceしてからscaleを求める。新経路もこの**reduce後の最大絶対値**を保存・供給し、4値の一部だけを使う不具合を避けた。SDKの各部分値そのものを再現する必要はなく、消費側が使用するglobal maxを一致させている。

SDK最小実験ではPD0/PD1ともに、±0、FP16最小subnormal、subnormal/normal境界、丸め境界、±65504、Inf、NaNを確認した。有限FP16の絶対値bitをunsigned比較し、最後にFP32へ戻すとSDK maxに一致する。NaNの扱いを一般化して採用せず、通常writerの例外flagまたはverify時の再検証で非有限値を検出した場合はSDKへ戻す。

findmaxが追加で読む論理量は1 APIあたり `4 × D × HK × S` byte。D64/HK4/S4096で4 MiB、D128/HK32/S4096で64 MiBである。これは**findmaxによる追加走査**の量であり、SDNN KQ/KQV本体のK/V読み込みは別に残る。

追加waitありの段階別中央値（ms）。通常API/tokens/sには使用しない。cache列はCPU管理とK/V部分最大値の統合kernelを含む。

|PD|D|S|旧 K findmax|旧 V findmax|新 増分max/metadata|旧K+V / 新|
|---|---|---|---|---|---|---|
|0|128|512|0.055025|0.055052|0.013068|8.4×|
|0|128|2048|0.190696|0.190477|0.012980|29.4×|
|0|128|4096|0.371446|0.370408|0.013008|57.0×|
|0|64|8192|0.055019|0.055024|0.013216|8.3×|
|1|128|512|0.055279|0.055084|0.012664|8.7×|
|1|128|2048|0.190336|0.189603|0.012875|29.5×|
|1|128|4096|0.369674|0.370124|0.012781|57.9×|
|1|64|8192|0.056228|0.056003|0.012866|8.7×|

D128/S4096ではfindmaxがAPI時間の約2割を占め、これを完全に無くしてもAPIの上限は概ね1.24倍。実測のAPI改善と整合する。SDK内部の非公開DMA命令数ではなく、API呼び出し数と論理走査byteを数えた。

## 3. KV更新経路と整合性

追跡対象はbackend所有の、連続FP16・single streamの `cache_*k_l*` / `cache_*v_l*` root。1物理token行のfeature数は `D × HK` で、GQAの全KV headを含む。現在のSDK契約がK/Vそれぞれのglobal maxなので、head別scaleへ変更していない。

管理情報はdevice ID、所有root/range、feature数、capacity、generation、revision、dirty区間、行のzero/finite証明、cached viewの開始位置・長さ、pending writer状態。pointerの一致だけではhitにしない。最大値はHBMのprivate Bufferへ保持し、L3契約を変更していない。

| 更新経路 | 検知・処理 |
|---|---|
| 通常SET_ROWS | 既存のCPU index検証とGPU bounds検証を維持。完全な物理行を所有する場合だけ専用writerへdispatch |
| CPU H2D / D2D / memset / state read | `kunlun::copy`で対象byte範囲をdirty化し、zero/finite証明とglobal maxの有効性を落とす |
| PD間copy | `peer`で宛先範囲をinvalidate |
| backend graphの書き込み | view等のread-only opと専用SET_ROWSを除く出力範囲を事前invalidate |
| buffer clear(0) | 実データのzero完了後にzero/finite証明とmetadataを初期化 |
| release / 再確保 | 重なる登録をretire。新しいgenerationで登録し、古いownerのdestructorから新ownerを消さない |
| 不明なraw SDK書き込み | 登録対象にしない。登録APIの利用者は明示invalidateが必要。標準backendの更新経路は上記hookを通る |

同じrootをKとVの両方に使うaliasは、private global maxの所有を混同しないようSDK fallbackとする。multi-stream root、非compact view、gather後の一時buffer、未登録tensorもfallbackである。

writer終了とstatus取得は既存と同じ同期境界を維持し、dataとmetadataを途中で参照しない。pending部分最大値はrootごとのBufferに分離した。shared engineのdescriptorは前のwriter完了後に再利用する。Flash APIは既存通り完了を待って戻るため、metadata/入力/workspaceの寿命はその完了まで保持される。

## 4. 採用方式

### writer

TGは従来と同じ32 workerのfeature並列cast/writeを保ち、**格納したFP16値**から各workerの最大絶対値を求める。32個の部分最大値はHBMに置き、次のAttention前処理でK/Vをまとめて1 kernelで統合する。最大値のD2H転送はない。

PPは全行maxの算出を省き、元のFP32→FP16変換のoverflow/NaN/Inf分岐でstatusを立てる。通常有限値では元の変換手順を維持する。FP16 sourceも格納値の有限性を確認する。将来cacheを使えるcapacityを持つ短contextのTGもmaxを求めずこのchecked writerを使用し、後でcontextが伸びた際に有限性の証明を失わないようにする。

statusは既存1回のD2Hを4→8 byteに拡張した。indicesのD2H、descriptor転送、writer launch/wait回数は増やしていない。PP/短contextではrow max/global max Bufferを確保しない。capacity自体が閾値未満ならwriterも従来経路を使用する。feature数256未満は測定範囲が不足するためautoでは有効化しない。

### SDK seedと増分更新

初回、view変更、上書き等ではSDK findmaxを実行し、4部分値の最大をGPUでprivate globalへ保存する。SDK値は有限FP16をFP32で表した値なので、FP16 magnitude bitへの格納は可逆である。

次回に以前の物理viewがprefixとして残り、新しい書き込みが証明済みzero行なら、`max(old_max, new_row_max)` を計算する。viewが伸びた場合は、新しく増えた範囲の有効row metadataだけを読む。未計算のdirty metadataがその新範囲に含まれればSDKでseedし直す。

実データを変えないlogicalな削除だけでは、物理KVを勝手にゼロ扱いしない。paddingや削除済みの物理値を旧SDKが走査するなら、その値を含めて同じmaxを維持する。viewの縮小・offset変更時は必要な範囲のSDK maxを取り直す。

### fallback / verify

- `KUNLUN_FLASH_KV_MAX=legacy`: 従来の全量SDK findmax。
- `auto`（既定）: 上記の安全な条件だけで再利用。大きいfeatureではS512以上、D64/HK4等の小さいfeatureではS8192以上を基本とする。未検証の書き込みやrestore後はSDK fallback。
- `verify`: cache hitごとに独立したSDK findmaxとbit比較。dirtyな有限性不明データはCLで再検証できる。差が出たらlayer名、generation/revision、view、4部分値、直近16更新を出力して停止する。

調整用に `KUNLUN_FLASH_KV_MAX_MIN_S`、比較用に `KUNLUN_FLASH_KV_MAX_REBUILD=rows` / `KUNLUN_FLASH_KV_MAX_REDUCE=linear` / `KUNLUN_FLASH_KV_MAX_PREFILL=fused` / `KUNLUN_FLASH_KV_MAX_WRITER=lazy` を残した。これらは通常測定では使用しない。`KUNLUN_ATTENTION_MATMUL`、Flash-Decoding、KQ/Online/PV等の既存設定は維持した。

## 5. 検討して採用しなかった方式

| 方式 | 結果・判断 |
|---|---|
| TGを1行/1 coreのwriterに融合 | 元のfeature並列性を失い、維持費用が大きかった。TGは32 workerに戻した |
| 毎回row metadata全体をscalar reduce | KV全量読み込みは減るが、小featureではCL reductionがSDKより遅い。global prefixを再利用する方式へ変更 |
| PPで全行maxを融合 | 7BのPPが約4%悪化。PPは有限性確認だけにした |
| PPを旧writerへ戻し、最初のTGでCL全量再構築 | 31 TGの評価でも7Bが約18%悪化。通常経路はSDK seedとし、CL再構築はverify/比較用に限定 |
| pointerだけのprefix max / 単純monotonic max | overwrite・restore・slot再利用・paddingで不正確になるため不採用 |
| metadataをpersistent L3へ配置 | 追加量は小さく、Q4/Flash workspaceとの寿命競合を増やす利益がないためHBM private Bufferを採用 |

## 6. カーネル/API性能

`KV_PERF`はKV不変時のsteady API、`KV_APPEND`は**K writer + V writer + Attention API**を実測した値。append測定は同じ有限prefixとzeroの未使用slotを準備し、16回の追加の中央値を取る。入力/indicesの準備uploadは単体timer外。LLM測定にはその実際の更新・管理費用が含まれる。

|PD|D/HK|S|steady API ms 旧→新|2 writer+API ms 旧→新|append速度比|
|---|---|---|---|---|---|
|0|64/4|4096|0.465 → 0.464|0.554 → 0.554|1.000×|
|0|64/4|8192|0.906 → 0.824|0.997 → 0.932|1.070×|
|0|64/4|16384|1.777 → 1.607|1.858 → 1.698|1.095×|
|0|128/32|512|0.535 → 0.425|0.620 → 0.530|1.170×|
|0|128/32|2048|1.906 → 1.542|1.991 → 1.649|1.207×|
|0|128/32|4096|3.807 → 3.057|3.866 → 3.160|1.223×|
|0|128/32|16384|14.977 → 12.533|15.382 → 12.611|1.220×|
|1|64/4|4096|0.458 → 0.458|0.536 → 0.537|0.998×|
|1|64/4|8192|0.900 → 0.819|0.990 → 0.925|1.070×|
|1|64/4|16384|1.763 → 1.606|1.848 → 1.693|1.092×|
|1|128/32|512|0.503 → 0.407|0.585 → 0.513|1.140×|
|1|128/32|2048|1.843 → 1.479|1.926 → 1.583|1.216×|
|1|128/32|4096|3.660 → 2.936|3.735 → 3.032|1.232×|
|1|128/32|16384|14.482 → 12.073|14.932 → 11.992|1.245×|

D64/HK4/S4096は最終auto fallbackの値。それ以上とD128/HK32は採用したcache経路の値。on/offのfull grid、D64/HK32およびD128/HK4の交差shapeも保存した。

全測定はD64/128、S256/512/1024/2048/4096/8192/16384、PD0/PD1、Flash-Decoding off/onを含む。full gridと各試験の環境・elapsed・device stateは保存ログにある。

## 7. 実LLM性能

保存した0.7.3 buildとの比較。prefix長は入力token数、TGは続く31 step（4096は63 step）のprocess時間から算出。PP/TGを別々に評価し、診断waitを入れていない。短contextではmax cacheを選ばないため、差が小さい値は測定変動を含む。

|model|PD|prefix|PP t/s 旧→新|TG t/s 旧→新|TG変化|
|---|---|---|---|---|---|
|tiny|0|128|279.55 → 281.46|22.61 → 22.61|-0.00%|
|tiny|0|512|278.08 → 290.55|21.91 → 20.97|-4.28%|
|tiny|0|1024|267.29 → 272.03|21.03 → 20.98|-0.23%|
|tiny|0|2048|248.95 → 254.87|20.48 → 19.58|-4.37%|
|tiny|1|128|281.36 → 280.86|22.71 → 22.12|-2.59%|
|tiny|1|512|281.30 → 278.80|22.00 → 21.80|-0.93%|
|tiny|1|1024|269.02 → 268.40|20.66 → 20.80|+0.70%|
|tiny|1|2048|250.43 → 252.21|20.57 → 19.08|-7.25%|
|7b|0|128|70.86 → 71.73|7.01 → 7.02|+0.07%|
|7b|0|512|70.76 → 70.90|6.38 → 6.46|+1.36%|
|7b|0|1024|69.18 → 69.51|5.76 → 5.98|+3.88%|
|7b|0|2048|67.05 → 67.57|4.91 → 5.16|+5.24%|
|7b|1|128|70.60 → 70.90|7.27 → 7.29|+0.19%|
|7b|1|512|70.36 → 70.61|6.58 → 6.74|+2.51%|
|7b|1|1024|69.09 → 69.66|5.99 → 6.19|+3.34%|
|7b|1|2048|66.72 → 67.23|5.07 → 5.40|+6.43%|
|tiny|0|4096|218.19 → 218.01|18.76 → 18.70|-0.33%|
|7b|0|4096|62.18 → 62.38|3.77 → 4.12|+9.53%|
|tiny|0|8192|175.39 → 175.55|14.32 → 14.57|+1.74%|
|tiny|1|8192|178.74 → 175.95|13.96 → 14.74|+5.63%|

各最終組合せは1 pairの測定。TGは31/63/127 stepの合計時間で、差の小さい値にはrun間変動が含まれる。Tiny 4096はcacheを採用せずSDK fallback、8192で有効にした。PD1/Tiny8192の+5%台すべてをfindmaxだけの効果とは断定しない。

Split（prefix512、15 TG）:

|model|split|PP t/s 旧→新|TG t/s 旧→新|TG変化|
|---|---|---|---|---|
|tiny|layer|277.44 → 277.49|21.98 → 21.96|-0.11%|
|tiny|row|169.62 → 169.26|13.00 → 13.12|+0.92%|
|7b|layer|70.61 → 70.33|6.48 → 6.57|+1.37%|
|7b|row|46.03 → 46.11|4.65 → 4.73|+1.68%|

Flash-Decoding無効時:

|model|decoding|prefix|TG t/s 旧→新|TG変化|
|---|---|---|---|---|
|tiny|off|2048|19.74 → 19.72|-0.13%|
|7b|off|2048|4.95 → 5.14|+3.86%|

## 8. メモリ・転送・起動回数

通常のsteady appendではK/V findmaxの追加全量走査2回が0回になる。代わりにCL max更新1 launchを行う。seed時だけ従来SDK 2回とcapture 1 launchが必要。writerは従来と同じ2 launchで、元からある同期境界を維持する。

FP16 cacheをもう一度HBMから読み直さず、writer LM上の格納FP16から部分最大値を作る。通常の固定viewでは、最大値更新の論理readはK/V合計約272 byte（部分値・前row値・global値）。contextのpadding範囲が伸びると有効row metadataのreadが加わる。KQ/KQV自身のDMAは削減量に含めない。

rootごとのdevice metadataは `4 × capacity + 512` byte（row maximaと256 byte aligned global/partial）。7B/ctx4096/64 rootでは1,081,344 byte、Tiny/ctx8704/44 rootでは1,554,432 byte。別途PD共有descriptor 512 byteがある。verifyのreference/jobsは通常autoでは使用しない。CPU側zero/finite flagは2 byte/物理row。新しいL3 workspaceは確保しない。

|kernel|最大LM frame byte|上限|
|---|---|---|
|kunlun-kv-max-build|704|16256|
|kunlun-kv-max-reduce|1088|16256|
|kunlun-kv-max-writer|3136|16256|

通常TGではmax値のD2H、追加の明示wait、metadataの毎回H2D uploadはない。行末の部分最大値は既存cast/writeと同時に生成し、CL kernel 1回でK/Vをまとめて供給する。

## 9. 精度・生成品質・互換性

|model|batch|ctx|score数|旧PPL|新PPL|logits NMSE|平均KL|
|---|---|---|---|---|---|---|---|
|tiny|128|256|1020|11.1246713081|11.1246713081|0|0|
|tiny|1|512|511|7.79196468091|7.79196468091|0|0|
|7b|128|256|1020|8.34594649553|8.34594649553|0|0|
|7b|1|512|511|5.68161902336|5.68161902336|0|0|

TinyのB1は品質専用 `MIN_S=256` でもactive cacheを追加評価し、511位置すべてでlogits bit一致、NMSE/KL/max_abs=0、PPL=7.79196468091。通常短context性能にはこのoverrideを使用していない。

合成単体のCPU FP64比較（Qを既存経路と同じF16へ丸めた参照）の最大NMSE=1.00064e-05、最大絶対誤差=9.54177e-05。この差は旧SDNNにもある量子化差で、新旧出力のbit差は0。contract 88 shape/設定の基本・更新試験、56 gridと交差shape、alias、特殊値、生成、restoreの成功ログを保存した。

最大値はFP16に丸めた**後**の値で求める。maxは有限FP16 magnitudeの整数比較なので、加算順序の変化や近似expを導入していない。seedはSDKの4部分値全体から求め、供給するglobal max/scaleを一致させる。合成AttentionのCPU FP64との差には、もともとのQ FP16化・INT16 MAC・確率FP16化等の誤差が含まれる。今回の変更による旧Attentionとの差は別にbit比較した。

append、最大値の上書き削除、lazy writer、部分H2D、同pointerの異データ、restore、範囲の縮小/拡張/offset、zero、FP16 subnormal/最大値、非有限値poison、F16/F32 source writerを試験した。KV入力・guardの不変性、GQA、head/batch broadcast、causal/all-mask、ALiBi、F16/F32 mask、stride/gather、端数queryを含む。Layer/Row Split、model再load、state restore後の次logits一致も確認した。非Flash Attention（転置Vを含む）も旧logitsとbit一致した。

**既存制約**: F16 KVに対するcontext shiftのin-place RoPEは、従来経路でもschedulerのsupports_opチェックで停止する。今回RoPEは変更していないため、この操作の成功は主張しない。max管理側には一般の書き込みinvalidateを用意した。M128/B2/D128の一部直接API試験も既存16 MiB workspace制約で拒否されるため、対応範囲内のM64/B2を試験した。

## 10. 運用記録と次の改善

実機のdevice state error・hang・resetは発生していない。初期の入力fixture長/context不一致、workspace上限の正常拒否、回帰試験のopt-in設定漏れ、および既存F16 RoPE scheduler停止は、原因・結果・ログを保存して条件を修正した。GPU異常とは区別した。

残る主な費用はSDNN K/V本体のDMA/MAC、FFN/Q4、KV writerに元からあるindices D2H・descriptor転送・完了待ち。任意overwrite/restore後に高速化を再開するには、PPで低負荷に作れるpage最大値/有限性metadata、またはstateに対応した安全なmetadata再初期化が候補になる。今回は不確かな経路をSDKへ戻し、精度と通常appendの性能を優先した。

既存70 kernelのbin SHA256は0.7.3とすべて一致。新規はwriter/build/reduceの3本。standalone/llama内蔵kernelも一致し、`git diff --check`は通過した。runtime `version()`とCMakeを0.7.4へ揃え、system libraryの必要版も0.7.4に更新した。installed libraryからのnative kernel読込とsystem backend buildも確認した。非Flash Attention（転置Vを含む）も旧logitsとbit一致した。

再現手順は `REPRODUCE.ja.md`、動作ソース・patch・raw log・集計値・入力fixtureは同梱artifactと実機 `results/kv-max-20261026` に保存した。
