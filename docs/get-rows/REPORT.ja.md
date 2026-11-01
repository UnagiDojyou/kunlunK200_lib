# Kunlun K200 Q4_0 Embedding GET_ROWS XPU実装

2026-11-01 JST。**Embeddingの参照・逆量子化をK200上へ移し、CPU GET_ROWSとEmbedding出力のH2Dを解消した。数値はCPUとbitwise一致したが、LLM全体の明確な高速化には至らず、既定OFFのopt-inとして残す。**

## 基準とCPU fallbackの原因

別チャットのQ6オンチップ版を固定した。開始時は両repoともclean。

| repo | 開始HEAD |
|---|---|
| library | 31cd7d76426381bc36aa19f121b39e8b119d9595 |
| llama.cpp | ebb192c9329ccafe86f70cf8b11ab049226174c9 |

独立branch/worktreeはcodex/k200-embedding-20261101。既存0.7.12、Q6新旧、CPU経路は保護した。Q6=1/LEGACY=0、SwiGLU=1、Large PP=0を固定し、common Q4、Attention/Flash/RoPE、Activation共有、stream、N-major、Ownersを維持した。実LLMログのQ6 onchip_callsとjoin_launches=0を全比較で確認した。

CPU fallbackには2つの理由があった。ggml-kunlunがGET_ROWS未対応であり、さらにllama-model.cppが入力層をCPUへ固定していた。supports_op追加だけでは重みがXPUへ移らないため、明示ONかつ最初の層がKUNLUNにある場合だけ入力層の候補を変更した。ローダーの支持判定用size=0仮bufferにはbuffer_contextがない点も対応した。

| 実GGUF token_embd.weight | type | [K,N] | 元Q4行bytes | GPU常駐payload |
|---|---|---|---:|---:|
| Llama 2 7B Q4_0 | Q4_0 (2) | [4096,32000] | 2304 | 73,728,000 B |
| TinyLlama Q4_0 | Q4_0 (2) | [2048,32000] | 1152 | 36,864,000 B |

出力層は別のQ6_K (14) tensorで、今回の2モデルはtied embeddingではない。入力はI32 token ID vector、出力はcontiguous F32 [K,M]。以前はCPUで出力を生成し、後続RMSNorm/TransformerへH2Dしていた。

## 実装・メモリ・互換性

共通Q4 ABI v1のN32×K32・16-bit bank-major W4、K-major FP16 scaleをそのまま利用する。特定行の隣接2 bankを64 byteで読み、signed nibbleを復元し、元K32 scaleを掛ける。元GGML Q4_0のoffset 8・low/high K順序は既存packで正確に再配置されており、weightの再量子化はない。MACは不要なのでCLを使用した。

- libraryにcommon_q4_get_rows API、CL kernel、既存Kernel内のProgram/status管理を追加した。GGML側に逆量子化処理を重複実装していない。
- ABI/layout/scale order、PD、ready、byte数、alignment、非aliasを検査する。全IDを検証してから参照し、無効IDでは出力を書かない。runtime errorは既存fail-stopを維持する。
- CommonQ4WeightをGGML model buffer内で借用し、追加のraw/F16/F32 HBM weightコピーは持たない。1 tensor/owning PDの常駐領域・upload counterは1。PP128とTG1のログで同じweight pointerとuploads=1を確認した。
- 同じGGML tensorを共有する場合は既存handleを再利用する。tied Row Splitは追加の全表copyを避けるためCPU入力を維持する。未知のtiedモデルでの実LLM評価は行っていない。
- GGML支持条件はQ4_0 matrix、common ABI、K%64=0、contiguous I32 IDs M=1..512、contiguous F32 output。weight view、他dtype/layout、strided/batched indices、M>512はCPUへ戻す。
- ID更新は毎回読み、pointerだけのcache判定をしない。statusは256 B、kernelは初回load後再利用。L3追加は0。Q4/Q6/Flash/SwiGLU/SET_ROWSの既存演算kernelとQ4 ABI・Leaseは変更していない。

主な変更はlibraryのget-rows.inc、kunlun-get-rows.xpu、API/CMake/stack guard/test、およびGGML backend、入力層配置、schedulerの限定的copy条件である。

### 非選択分岐の修正

診断で、未使用のmixed-token GET_ROWSが未初期化IDで実行され、ID guardによりhost abortした。両PDはRUNNING/no error、errtask不変で、device errorやresetはなかった。

CPU実装と同様に、GET_ROWSはGGML_TENSOR_FLAG_COMPUTEを尊重するよう修正した。さらにschedulerが非選択GET_ROWSの未使用出力をCPUへcopyしていたため、KUNLUNのそのcopyだけを省略した。既存のsource/destination/event待ちは維持し、fenceは削除していない。診断callbackも非選択nodeを集計しない。無効IDを持つ非選択分岐が実行されず、出力guardを保持する回帰テストが合格した。

停止ラッチを無条件削除せず、host側原因・修正・両PD状態を記録して履歴へ移し、小規模テストから再開した。修正前の性能値は最終比較に使用していない。

## 単体GET_ROWS

実Embeddingの全32000行を使用。PD0、CPUは持続GGML threadpool 6 threadのGET_ROWS、XPUはweight upload/kernel loadを除いたAPI完了時間。warm後3回中央値、ms。初期のOpenMP参照起動費入り時間は採用せず、以下は実GGML operatorを用いた値である。

| モデル | M | CPU | XPU | CPU＋出力H2D | ID H2D＋XPU |
|---|---:|---:|---:|---:|---:|
| 7B | 1 | 0.0011 | 0.0227 | 0.0121 | 0.0609 |
| 7B | 128 | 0.1023 | 0.9964 | 0.4736 | 1.0028 |
| 7B | 512 | 0.4254 | 3.8385 | 1.7097 | 3.8456 |
| Tiny | 1 | 0.0010 | 0.0567 | 0.0213 | 0.0609 |
| Tiny | 128 | 0.0496 | 0.5539 | 0.2685 | 0.5612 |
| Tiny | 512 | 0.2319 | 1.9451 | 0.9256 | 1.9641 |

各列は別系列の中央値で、加算できない。特にM1は起動・waitのばらつきが大きい。XPU列にはID validationのstatus D2H 256 Bと完了待ちを含め、精度確認用のActivation全体D2Hは含めていない。**単体ではCPU＋転送よりも遅い。**

試した改善は64/256要素jobの比較と、非選択分岐の実行・copy除去。256要素jobはPPの出力DMA回数を減らし、M512で7B 4.101→3.833 ms、Tiny 2.075→1.946 msだった。M1は64要素jobで並列性を確保する。大きなcacheや非同期pipelineは追加していない。共通packの行アクセスではK32あたりweight/scaleの小DMAが残り、これが単体の主な制約である。

## 精度・動作

| 確認 | 結果 |
|---|---|
| CPU Q4 dequant / GET_ROWS一致 | 両モデルbitwise一致 |
| M=1/3/128/512、重複・順不同・ID0/N−1 | 合格 |
| invalid ID −1/N、短いbuffer、M513 | 拒否、前後guard保持 |
| 全出力finite・全要素書込み | 合格 |
| PP512後から8位置のlogits | 両モデルbitwise一致、NMSE/max abs/KL=0 |
| greedy128 | 両モデル128/128 token一致、大量logits dumpなし |
| F16 KV保存・復元、PP→TG | 復元後next logitsが各構成内でbitwise一致 |
| PD1 / Layer / Row Split | Tiny PP128→TG4合格 |
| 非選択分岐・flag OFF/未設定・unsupported descriptor | 合格 |

PD1では独立libraryテストのM1/3/128/512もbitwise一致した。Row Split時のEmbeddingはnormal PD0に1個、Q6出力はN16000ずつ両PDで新オンチップ版が動いた。全Splitの性能測定、LoRA/mixed入力の実LLM品質、全未知モデルの評価は追加していない。数値が完全一致したためPPLの再測定は不要と判断した。

## 通常LLM性能

標準llama-bench PP512/TG128、Single PD0、threads6、batch/ubatch512/128、F16 KV。要求contextは標準benchのn_prompt+n_gen+n_depth（PP512、空KV TG128）、runtime paddingも標準動作。前述の新Q6/SwiGLU/Attention等を固定した。同一binary/libraryでGET_ROWSだけ0→1→1→0、各arm n=2、平均＝中央値±標本SD。observer、強制同期、状態保存はこの測定に含めない。

| モデル | phase | CPU Embedding token/s | XPU Embedding token/s | 変化 |
|---|---|---:|---:|---:|
| 7B | PP512 | 80.210 ±0.019 | 80.301 ±0.057 | +0.11% |
| 7B | TG128 | 8.545 ±0.010 | 8.520 ±0.018 | −0.29% |
| Tiny | PP512 | 317.261 ±0.142 | 318.281 ±0.541 | +0.32% |
| Tiny | TG128 | 28.863 ±0.018 | 28.784 ±0.001 | −0.27% |

大きな回帰はないが、少数標本の小差であり、明確な総合高速化とは判断しない。CPU Embeddingはもともと短く、DMA/launch/status確認と残存graph分割の費用がある。

## 転送・CPU fallback・HBM

以下は通常速度とは別のPP128→TG4診断、context1024。TGは実decode3回の合計を3で割った平均。SDK copy呼出しのinclusive時間であり、純粋なPCIe DMA時間ではない。kernel/待ち時間へ加算しない。

| 指標 | 7B CPU→XPU Embedding | Tiny CPU→XPU Embedding |
|---|---:|---:|
| CPU Q4 GET_ROWS（PP / TG3回） | 1/3 → 0/0 | 1/3 → 0/0 |
| XPU Q4 GET_ROWS（PP / TG3回） | 0/0 → 1/3 | 0/0 → 1/3 |
| PP H2D bytes | 2,317,084 → 223,772 | 1,252,316 → 207,580 |
| PP D2H bytes | 4,421,120 → 4,421,376 | 2,292,912 → 2,293,168 |
| TG H2D bytes/token | 82,664 → 66,288 | 63,869.3 → 55,685.3 |
| TG D2H bytes/token | 162,048 → 162,304 | 145,088 → 145,344 |
| PP H2D API ms | 1.184 → 0.823 | 1.557 → 1.375 |
| PP D2H API ms | 2.595 → 2.656 | 1.767 → 1.974 |
| TG H2D API ms/token | 0.547 → 0.524 | 1.088 → 1.114 |
| TG D2H API ms/token | 1.482 → 1.482 | 1.018 → 1.034 |
| Single PD P2P | 0 → 0 | 0 → 0 |
| reserve graph分割数 | 4 → 5 | 4 → 5 |
| API-tracked HBM peak (decimal MB) | 4452.740 → 4526.472 | 744.839 → 781.706 |

Embeddingの大きな出力H2Dが消えた。PP128で削減する出力本体は7B 2 MiB/Tiny 1 MiB、PP512では論理上8 MiB/4 MiB。ID転送と初回kernel load等は残る。D2Hの増分は安全検証用statusの256 B/呼出しだけで、非選択GET_ROWSの大きな読戻しは除去できた。

**graph分割そのものは減らず、4→5となった。** 非選択分岐のsplit計画・同期は残す保守的実装である。データcopyは省いたが、splitやwaitを無検証で削除していない。CPU F32 GET_ROWSはPP2回/TG6回残り、7Bではattn_out-31/l_out-30の各2 MiB（このPP128診断）を読戻す。これとKV index取得等が残るCPU処理である。

HBM増分はEmbedding圧縮payloadに加え、code/status等の割当差が7Bで3,584 B、Tinyで3,072 Bだった。新しい恒常的なF16/F32 weight展開はない。CPU pack/uploadは7B約200/11 ms、Tiny約100/5.6 msの一度だけ。モデル全体load時間のcold-cache変動と混同しない。HBM値はSDK allocation hookのpeakで、CPU RAMや全driver内部領域を含む値ではない。

## 採否・残る候補・安全性

GGML_KUNLUN_GET_ROWS=1でXPU実行可能、未設定/0/無効値はCPUのまま。精度とXPU集約は達成したが、単体は遅く、TGも約0.3%低下を観測したため既定化しない。Q6のデフォルト、SwiGLU、Large PPは変更しない。

次はF32 GET_ROWS（out_idsによる最終hidden選択）のXPU化、非選択分岐をsplit計画から外す際のgraph再利用条件の検証、必要ならEmbedding行アクセスに適したlayoutを「二重常駐なし・tied weight安全」の条件で検討する。共通packのDMA効率だけを追って大きなメモリcacheは追加していない。

thermal guard 80℃、persistent latch、PD/他利用者確認を継承。最大観測60℃、両PD RUNNING/no error、errtask不変。host abort後は原因を調べた上でレビュー履歴を残し、小テストから再開した。reset・driver/firmware/clock/OS変更はない。モデル・SDKの複製、巨大log、corpus評価は行わず、空き容量は約36.7 GBを維持した。

## 再現とGit

実機R=/home/unagidojyou/kunlun-k200。worktreeはR/worktrees/embedding-20261101-libと-llama、buildも同名、install/logs/小さいJSONはR/results/embedding-20261101。

機能commitはlibrary bcaad3e745e8b26282df292737b7392c334ae605、llama.cpp 95ca2308fc5080dcbeb63f8ea3d5958f840c126d。最終の文書commit/HEAD/gitlinkはdelivery-summary.jsonに記録する。新APIを含むlibraryとbackendを揃えること。GitHub push・PR・main/master更新は行っていない。

~~~sh
R=/home/unagidojyou/kunlun-k200
O=$R/results/embedding-20261101
L=$R/worktrees/embedding-20261101-lib
P=$R/worktrees/embedding-20261101-llama
# SDK環境 XRE / XDNN / XTDK_DIR を既存値のまま使う。
cmake -S "$L" -B "$R/build/embedding-20261101-lib" \
  -DCMAKE_BUILD_TYPE=Release -DKUNLUN_BUILD_BENCHMARKS=OFF \
  -DCMAKE_INSTALL_PREFIX="$O/install"
cmake --build "$R/build/embedding-20261101-lib" -j4
cmake --install "$R/build/embedding-20261101-lib"
# 既存buildは変更していないkernelだけprebuiltを利用。
# source-onlyでは -DKUNLUN_PREBUILT_KERNELS= -DKUNLUN_XTDK="$XTDK_DIR" を指定。
cmake -S "$P" -B "$R/build/embedding-20261101-llama" \
  -DGGML_KUNLUN=ON -DKUNLUN_USE_SYSTEM_LIBRARY=ON \
  -DCMAKE_PREFIX_PATH="$O/install" -DLLAMA_BUILD_TOOLS=ON
cmake --build "$R/build/embedding-20261101-llama" -j4 \
  --target llama-bench kunlun-llm-smoke \
  kunlun-get-rows-policy-test kunlun-get-rows-inactive-test
export GGML_KUNLUN_COMMON_Q4=1 GGML_KUNLUN_SWIGLU=1 KUNLUN_LARGE_PP=0
export KUNLUN_Q6_K_SDNN=1 KUNLUN_Q6_K_LEGACY=0
export GGML_KUNLUN_GET_ROWS=1       # 0でCPU Embeddingへ戻す
# 実機実行はthermal guard / latch / PD確認を行う既存runner経由。
python3 "$O/bench.py"             # GET_ROWSだけ0/1 ABBA、両モデル
python3 "$O/smoke.py"             # PP512 logits / KV保存復元
python3 "$O/greedy128.py"         # NO_DUMPで128token比較
python3 "$O/ggml-unit.py"         # 持続CPU poolを使う単体比較
python3 "$O/regression.py"        # PD1 / Layer / Rowの小規模確認
~~~

対応条件と64/256要素の検証用overrideはGET_ROWS.mdに記載した。各試験の正確なargv・環境はresults/logs/*.jsonに残す。性能評価は最終修正後のABBAだけを採用し、初期試作や診断値と混ぜない。
