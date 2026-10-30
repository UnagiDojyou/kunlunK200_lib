# Kunlun K200 Q6_K SDNN GEMV 開発結果

2026-10-30。0.7.12を基準に、Q6_K出力層をSDNNへ接続した。**XPU上での実行・精度検証は完了したが、通常推論のTG高速化は未達であり、既定OFFの実験経路として採用する。** CPU fallbackは維持する。GitHub push、PR、main/master更新は行っていない。

## 実装と基準

開始時は両worktreeともclean。libraryは c967bd151b8c19e8d3c46bb567e57d5f77a3f2a6、llama.cppは a6846395abfef34336773db66da467e338c90db3。前回CL実験と、要求済みのSwiGLU既定ONを含む状態から独立branch codex/k200-q6-sdnn-20261030 を作成した。元の安定版・build・CL試作は保護した。

Q6_KはGGML supports_opが未対応型として拒否していた。実GGUFのoutput.weightはtype14、N=32000、K=4096（7B）/2048（Tiny）。CPUではQ6_K×Q8_Kを計算していた。

- load時にsigned6値をlossless W8へ展開し、K16ごとのFP16 d×signed INT8 scaleをFP32で保存する。異なるscaleの整数積和を合算しない。
- 専用Q6KWeight/Q6KWorkspace/q6_k_matmulを追加。weightとworkspaceの寿命を分離し、ABI・ready・PD・shape・alignment・非aliasを検査する。1 tensor/PDあたりweight allocation/uploadは1回。PP→TGで再pack/uploadしない。
- 専用CLでchunk256 A8量子化し、SDNN W8A8 MAC、EW scale補正、FP32 partial、CL最終集約を実行する。ActivationのCPU readbackはない。
- Q4からProgram/Buffer/runtime_mutex、DMA/DS/EW/RS手順、8 controller専用化、N-major走査、L1D常駐設計を再利用した。Q4 ABI、Lease、既存量子化・GEMV/GEMM・Attention・SwiGLU・SET_ROWSは変更していない。
- 4 clusterでN32 tileを分担。DMA→DS→MAC→EW→RS→DMAOUTの完了・syncを維持する。L2/L1W/resultは2 buffer、scale L1Eは3 slot。MACの追加出力を別領域へ隔離し、既存のK16 partialを上書きしない。
- KL1 assemblyに実際の mm ...,16（SDKのsigned W8A8 mode）を確認した。Q6の追加L3確保はない。

変更ファイルはlibraryの include/kunlun/k200.h、src/q6-k.inc、src/k200.cpp、3つのkunlun-q6-*.xpu、CMake、bench/q6-k-test.cpp。GGML側はggml-kunlun.cppの配置・host shadow・転送・dispatch・Splitと、q6-policy-test.cpp。実験ソースはexperiments/q6-sdnnへ分離した。

## 段階的な改善

以下はSDNN partial生成だけの中央値（量子化と最終joinを含まない）。同一実重み、PD0、M=1、ウォーム後3回。

| 段階 | 7B ms | Tiny ms | 内容 |
|---|---:|---:|---|
| 逐次controller、K16をK32へzero pad | 124.956 | 62.502 | 正確性の初期基準 |
| 8 controller pipeline＋Activation常駐 | 5.495 | 2.781 | K16×16の一括DMA、N-major |
| K16直接MAC＋W8 zero pad削除 | 3.739 | 1.900 | 転送weight半減、MAC/DS短縮 |

小規模K256/N128および実出力層の**全K16 partialが同じ整数MAC＋scale CPU参照とbitwise一致**した。初期試作が遅い段階で停止せず、構造的に改善した。

方式B（圧縮Q6をXPUでtile展開）は未実装。今回は方式Aのload時展開で実行可能になったため、追加の展開pipelineは増やしていない。L1内FP32集約や最終join融合も未実装である。

## 単体性能・メモリ

固定実重み、x[j]=sin(j×0.123f)×0.4、PD0、CPU6 thread。定常実行の3回中央値。SDNN時間は**XPU量子化・SDNN・join・必要な完了待ちをすべて含み**、pack/uploadを除く。CL版は以前のFP32 Activation直接計算を変更せず再実行した。

| モデル・K×N | CPU Q8_K ms | CL F32 ms | SDNN A8 ms | SDNN対CPUの時間増加 |
|---|---:|---:|---:|---:|
| 7B・4096×32000 | 3.487 | 148.523 | 4.576 | +31.3% |
| Tiny・2048×32000 | 1.682 | 74.435 | 2.540 | +51.0% |

CL比では約32.5倍/29.3倍だが、CPUには届かない。新SDNNのmin/maxは7B 4.572/4.579 ms、Tiny 2.528/2.551 ms。最終整理後の独立library testでも7B 4.578 msを確認した。

| 項目（decimal MB） | 7B | Tiny |
|---|---:|---:|
| 元Q6_K bytes | 107.52 | 53.76 |
| W8＋FP32 scaleの常駐pack | 163.84 | 81.92 |
| HBM workspace | 33.031 | 16.516 |
| うちFP32 partial | 32.768 | 16.384 |
| CPU pack ms | 346.7 | 173.8 |
| pack H2D/upload ms | 24.4 | 12.3 |
| resident L1D / cluster | 256 KiB | 128 KiB |

W8 8bpw＋scale 2bpw＝10bpw。元Q6_Kの6.5625bpwに対し約1.52倍であり、元の圧縮率を維持した方式ではない。整数情報・scale精度は維持する。CPU経路ではこのweightはHBMにないため、CPU比では表のpack全量とworkspaceが追加される。host側にCPU fallback用の元GGML bytesを保持する。定常時のweight uploadは0、1 rowあたりkernel起動は3回。Q4 Leaseとは別bufferであり、既存Leaseの内容・世代・寿命を緩めていない。

load時間の単発参考値は7B 13.239→13.700秒、Tiny 2.081→2.309秒。反復したload benchmarkではなく、pack/upload以外の変動も含む。

## 精度・互換性

F64参照とCL版は元F32 Activation、新SDNNはchunk256 A8を使用する。このActivation量子化差と実装誤差を分離した。

| 指標 | 7B | Tiny |
|---|---:|---:|
| 元Q6×F32 F64参照へのNMSE | 1.0035e-5 | 8.9688e-6 |
| 同じA8値を使うF64参照へのNMSE | 8.9152e-14 | 3.9441e-14 |
| CPU Q6×Q8_KへのNMSE | 4.8575e-13 | 2.5925e-13 |
| 実CL出力へのNMSE | 1.0035e-5 | 8.9688e-6 |
| CPUへの最大絶対誤差 | 1.3113e-6 | 8.5682e-7 |
| CPUへの最大相対誤差（|ref|>0.01） | 9.4295e-5 | 4.3322e-5 |
| 元F32 F64参照への最大絶対誤差 | 0.0044224 | 0.0043797 |

signed6→W8、K16 scale対応、独立lane順decode、全出力finite、前後256-byte guard、ゼロ入力が合格。F16入力と、その同値F32入力の出力はbitwise一致。M=2でrow配置・ゼロ2行目・前後guardが合格。PD1でもTinyの単体試験が合格した。

PP512→greedy8 token、F16 KV保存/復元を両モデルでCPU/XPU切替比較した。

| LLM logits（8位置） | 7B | Tiny |
|---|---:|---:|
| NMSE | 8.2468e-14 | 3.2816e-14 |
| max abs | 9.5367e-6 | 8.5831e-6 |
| 平均KL | 3.3269e-12 | 4.1968e-13 |
| greedy一致 | 8/8 | 8/8 |

Logitsはbitwise一致しないが、CPUのQ8_K Activation経路との差はごく小さい。大規模PPL・task評価は追加しておらず、モデル品質の完全な同等性を証明したとはしない。

TinyのPD1、Layer Split、Row SplitでPP128→TG4とKV復元が合格。Row SplitはN16000ずつの両PDで実Q6 dispatchを確認した。Activation共有ON/OFFが動作し、復元後のnext logitsは各構成内でbitwise一致した。M>512、N32032、weight view、GGML F16 activation、環境変数OFF/未設定のsupports_op拒否をpolicy testで確認した。

## 通常LLM性能

標準llama-benchのPP512/TG128。TGは空KVからの独立試験。Single PD0、threads6、batch/ubatch512/128、F16 KV、SwiGLU=1、Large PP=0、common Q4/Attention/Flash/RoPE/Activation共有/stream/N-major/Owners有効。同一binary/libraryでQ6 flagだけを0→1→1→0に切替。各arm n=2、平均＝中央値、±標本標準偏差。callback、LD_PRELOAD、詳細profile、KV保存を含まない。

| モデル | phase | CPU経路 token/s | SDNN経路 token/s | 観測変化 |
|---|---|---:|---:|---:|
| 7B | PP512 | 80.514 ±0.231 | 80.616 ±0.300 | +0.13%（同等） |
| 7B | TG128 | 8.443 ±0.031 | 8.322 ±0.065 | −1.43% |
| Tiny | PP512 | 331.670 ±6.158 | 319.964 ±2.707 | −3.53% |
| Tiny | TG128 | 28.538 ±0.430 | 28.034 ±0.035 | −1.77% |

少数sampleであり、特にTinyのPP/TGにはばらつきがある。Tiny PP低下の全量をQ6 GEMVの費用だけで説明できたとはしない。追加の網羅測定は行わず、少なくとも高速化したとは判断しない。最終整理の変更はhost pack関数分離・コメント・終了時counter・policy targetであり、device演算は変更していない。

## CPU fallback・転送・graph

通常性能とは別に、7B PP128/TG4だけ既存audit-liteとlayout/node callbackで診断した。この同期付き時間を上記token/sへ混ぜていない。TG欄は実際のdecode3回の合計を3で割った値。

| 指標 | CPU経路 | Q6 SDNN |
|---|---:|---:|
| Q6 CPU MUL_MAT（PP / TG） | 1 / 3回 | 0 / 0回 |
| Q6 XPU MUL_MAT（PP / TG） | 0 / 0回 | 1 / 3回 |
| Q6 MUL_MAT TG / token（診断） | 3.587 ms | 4.674 ms |
| PP D2H | 4,309,504 B | 4,421,120 B |
| TG D2H / token | 50,432 B | 162,048 B |
| TG H2D / token | 82,664 B | 82,664 B |
| TG P2P | 0 | 0 |
| runtime D2H TG / token（診断・inclusive） | 1.437 ms | 1.465 ms |
| tracked HBM peak | 4,288,621,824 B | 4,485,505,536 B |

最終hidden stateの16,384 B/token読戻しは消える一方、CPU sampling用logits 128,000 B/tokenを新しく取得するため、D2Hは差引111,616 B/token増えた。転送回数や総graph分割が必ず減るわけではない。同期診断なし・state-check付きsmokeのXPU activation graph実行数は両モード23回で変わらず、Q4の共有group・GEMV/GEMM回数も同じだった。CPUのEmbedding/GET_ROWS、F32 SET_ROWS等は残る。

## 採否と次の改善

**KUNLUN_Q6_K_SDNN=1 の実験的opt-inとして統合し、未設定/0のCPUを維持する。** XPU集約という目的は達成したが、TG高速化とHBM削減は達成していない。SwiGLU既定ON、Large PP既定OFFも維持する。

残る優先候補は、MACの追加出力まで利用できるW8/DS配置、K16 FP32補正後のon-chip集約によるpartial/join削減、最後にXPU sampling/top-kによるlogits転送削減。方式BはHBM容量上必要になった場合に、tile展開費用と比較する。未検証のfence削除・reset・driver/firmware/clock変更は行っていない。

開発時には数値配置ミスを停止ラッチ付きで検出・修正した。MAC追加出力による上書きとscaleのbyte offsetを修正。ビルド失敗後に旧probeを一度使用したため、以後はビルド開始時に古いprobe binを除去した。新規device errorはなく、両PDのerrinfo/errtaskは開始時から不変、最大観測温度59℃。resetは実施していない。今回の失敗記録に重複保存された無関係な旧build snapshotのみ整理し、ログ・状態・試験binaryは保持した。

## 再現とGit

実機root R=/home/unagidojyou/kunlun-k200。

- library: R/worktrees/q6-sdnn-20261030-lib
- llama.cpp: R/worktrees/q6-sdnn-20261030-llama
- build: R/build/q6-sdnn-20261030-lib、R/build/q6-sdnn-20261030-llama
- install/logs/最小JSON: R/results/q6-sdnn-20261030

機能commit: library f9e5cb878c6076d8b5fd8ae28079ed241c11bf9c、llama.cpp 07f51fe854c68b6fe37eb24370ac2e1c91970d61。最終の文書commit、HEAD、gitlink、build-source情報はdelivery-summary.jsonに記録する。旧0.7.12のDSOには新APIがないため、version文字列だけでなく対応commitを揃えること。

~~~sh
# 既存SDKの環境変数 XRE / XDNN / XTDK_DIR を維持する。
R=/home/unagidojyou/kunlun-k200
L=$R/worktrees/q6-sdnn-20261030-lib
P=$R/worktrees/q6-sdnn-20261030-llama
O=$R/results/q6-sdnn-20261030
cmake -S "$L" -B "$R/build/q6-sdnn-20261030-lib" \
  -DCMAKE_BUILD_TYPE=Release -DKUNLUN_BUILD_BENCHMARKS=OFF \
  -DCMAKE_INSTALL_PREFIX="$O/install"
# 保存済みprebuiltを再利用する既存build。source-onlyの場合は
# -DKUNLUN_PREBUILT_KERNELS= -DKUNLUN_XTDK="$XTDK_DIR" を指定する。
cmake --build "$R/build/q6-sdnn-20261030-lib" -j4
cmake --install "$R/build/q6-sdnn-20261030-lib"
cmake -S "$P" -B "$R/build/q6-sdnn-20261030-llama" \
  -DGGML_KUNLUN=ON -DKUNLUN_USE_SYSTEM_LIBRARY=ON \
  -DCMAKE_PREFIX_PATH="$O/install" -DLLAMA_BUILD_TOOLS=ON
cmake --build "$R/build/q6-sdnn-20261030-llama" -j4 \
  --target llama-bench kunlun-llm-smoke kunlun-q6-policy-test
# 実機実行は既存thermal guard / latch / PD・利用者確認を行うrunner経由。
python3 "$O/bench.py"       # 同一binaryのQ6=0/1 ABBA、両モデルPP512/TG128
python3 "$O/smoke.py"       # PP→TG / KV保存復元
python3 "$O/contracts.py"   # F16 / M2 / guard / PD1
python3 "$O/diagnostic.py"  # 通常性能とは別の転送・配置診断
~~~

通常の選択方法は KUNLUN_Q6_K_SDNN=0（CPU）または1（SDNN）。他の基準flagはGGML_KUNLUN_SWIGLU=1、KUNLUN_LARGE_PP=0で固定する。詳細な再現argv・環境はresults/logs/*.jsonに保存した。SDK・モデル・build binary・大量traceをGitへ含めていない。
