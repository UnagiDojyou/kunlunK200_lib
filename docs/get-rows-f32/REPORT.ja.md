# Kunlun K200 F32 GET_ROWS XPU化

2026-11-01 JST。**最終層の2つのF32 GET_ROWSをCL/DMAでXPU化し、大きなActivationのreadbackと選択結果の再転送を解消した。CPUとbitwise一致し、graph分割は5から3へ減った。通常PP/TGの明確な高速化は確認できず、独立opt-inとして残す。**

## 基準・対象graph

開始HEADはlibrary a4f6f8869aabfcd34887460b5619a5c1f1327d5f、llama.cpp 945f7c5de8a229b09b810c58acd15309dca7d4b0。開始時は両方clean。独立branchはcodex/k200-getrows-f32-20261101、worktree/build/resultsもgetrows-f32-20261101名で作成し、正常なEmbedding版・0.7.12・Q6オンチップ版は変更していない。

全比較でcommon Q4、Attention/Flash/RoPE、Activation Lease、stream/N-major/Owners、SwiGLU=1、Large PP=0、Q4 Embedding GET_ROWS=1を固定。Q6はSDNN=1/LEGACY=0、onchip_calls>0 / join_launches=0を確認した。F32 GET_ROWSだけを0/1で切り替えた。

| モデル | 実際の入力 | PP128の入力shape | 出力 | index | 後続consumer |
|---|---|---|---|---|---|
| 7B | attn_out-31 | F32 [4096,128]、nb=[4,16384,2097152,2097152] | F32 [4096,1] | out_ids: I32 [1] | ffn_inp-31 のADD |
| 7B | l_out-30 | 同上 | 同上 | 同じout_ids | 同じADDの残差側 |
| Tiny | attn_out-21 / l_out-20 | F32 [2048,128]、nb=[4,8192,1048576,1048576] | F32 [2048,1] | out_ids: I32 [1] | 最終層ffn_inp-21 |

attn_outはAttention後のoutput projection、l_outは前層の残差・FFNを含む出力。src/models/llama.cppは最終層だけ両入力をout_idsで選択し、ADD、最終FFN、residual、output RMSNorm、Q6_K result_outputへ渡す。GET_ROWSから直接Q6へ渡すgraphではないが、その間を含めXPU上で連続実行できる。

out_idsはllm_graph_input_out_ids::set_inputがubatch.outputから生成するCPU上の入力。全tokenを出力する場合は0..M-1、通常の最後だけ出力するPP128ではローカル行127（PP512の最後のubatchなら絶対token511）、TGでは0になる。新実装はindexだけをH2Dし、両GET_ROWSで同じGPU copyを利用する。

これら2nodeはGGML_TENSOR_FLAG_COMPUTE=16であり、実際のselected node。以前の未使用mixed-token Q4 GET_ROWS node_1とは異なる。従来supports_opはGET_ROWSをcommon Q4 weight用にしか許可せず、CPUに配置されたため、schedulerが選択前Activation全体をD2H、選択後2行をH2Dしていた。

## 実装

- library: get_rows_f32 API、src/get-rows-f32.inc、kernels/kunlun-get-rows-f32.xpu。GGML型やgraph固有処理はlibraryへ持ち込んでいない。
- CL 4 cluster x16 coreで独立した出力tileを担当し、GM2LM/LM2GMで直接copy。unsigned 32-bit wordとして転送し、浮動小数点変換・MAC・再量子化はしない。
- M=1は64要素tile、複数行は1024要素tile。最終tileは長さを切り詰める。追加のActivation HBM bufferはない。
- 各workerが全indexを検証してから自身のデータを参照する。invalid IDでは全workerが出力を書かず終了する。検証status 256BだけD2Hし、kernel完了を待ってから判定する。
- ProgramとstatusをKernel内に保持し、既存mutex・PD binding・destructorで寿命を管理する。L3追加なし、CL frameは6400B/上限16256B。
- backendにGGML_KUNLUN_GET_ROWS_F32=1と専用Kernel/counterを追加。Q4 Embedding用GGML_KUNLUN_GET_ROWSとは独立。
- 対応は連続F32 matrix [K,N]、K%16=0、16<=K<=1048576、1<=N<=INT_MAX、連続I32 vector 1<=M<=512、連続F32 [K,M]。input/output view、特殊stride、batched indices、他dtype、split weight bufferは対応判定で拒否しCPUへ戻す。通常のLayer/Row SplitのActivationはnormal owning-PD bufferなので対応する。
- library APIはbyte数、64B input/output alignment、I32 alignment、address overflow、非overlapを検査する。active invalid IDはCPU版のassertと同様に安全停止し、無条件再実行しない。

GGML_TENSOR_FLAG_COMPUTEによる非選択GET_ROWSのskipと、非選択KUNLUN GET_ROWSのdata copyだけを省く前回のscheduler修正をそのまま利用した。今回scheduler本体、graph topology、fence、Lease、weight ABI、既存Q4/Q6/Attention/SwiGLU kernelは変更していない。

## 精度・安全性・互換性

| 確認 | 結果 |
|---|---|
| 独立memcpy参照と実GGML CPU GET_ROWS | bitwise一致 |
| K=2048/4096、M=1/3/128/512、K=16/80の端数tile | 合格 |
| 重複、順不同、ID0/N-1、不正ID -1/N | 正しい行取得 / 不正時は出力未変更 |
| 出力全領域・前後guard・入力不変 | 合格 |
| NaN payload、Inf、signed zero、subnormal | bit pattern保持 |
| 短いbyte数、alias、M513、unsupported view/stride/dtype | 拒否 / CPU配置維持 |
| 同じ選択済みgraph・indexアドレスを別内容で再利用 | 両PDで正しく更新、非選択invalid branchは未実行 |
| 両モデルPP512後から8位置のlogits・greedy8 | CPU/XPU dumpがbitwise一致、NMSE/max abs/KL=0 |
| PPからTG、F16 KV保存・復元 | 復元後next logitsが各構成内でbitwise一致 |
| Tiny PD1 / Layer / Row Split、PP128からTG4 | 合格。Layer/RowではF32選択がPD1、Q6新オンチップも正常 |
| Q4 Embedding、Activation共有 | uploads=1を維持、既存Q4共有group数も同じ |

品質差がないためPPL・大規模corpus評価は追加していない。全Split性能や未知のview/mixed/LoRAモデルを網羅したとは主張しない。

追加テストでggml_build_forward_selectをruntime分岐切替APIと誤解し、host assertが1回発生した。このAPIはgraph構築時の依存をmarkし、既存compute flagをclearしない。実装ではなくテストを修正し、同じ選択済みgraphのactive index更新を検証した。両PD RUNNING/no error・errtask不変を確認、失敗binary/source/dmesgとラッチをレビュー履歴へ保存し、小テストから再開した。resetはない。ホスト側のCMake target再生成・const型のコンパイル修正も完了した。

## 単体性能

PD0、CPUは既存GGML operatorと持続6-thread pool、warm後3回中央値、ms。weight uploadなし。GPUはindex検証、launch、完了待ち、status D2H込み。old_totalは入力全体D2H+CPU GET_ROWS+選択結果H2D、new_totalはindex H2D+XPU API。各列は別系列の中央値で、加算して再構成しない。

| K | N（入力行） | M（取得行） | CPU | XPU | 入力D2H | 出力H2D | index H2D | old total | new total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 1 | 1 | 0.0009 | 0.0559 | 0.0205 | 0.0079 | 0.0073 | 0.0159 | 0.0562 |
| 2048 | 128 | 1 | 0.0009 | 0.0541 | 0.2360 | 0.0079 | 0.0073 | 0.2153 | 0.0586 |
| 2048 | 128 | 128 | 0.0425 | 0.0717 | 0.2180 | 0.2405 | 0.0074 | 0.4534 | 0.0750 |
| 2048 | 512 | 512 | 0.1957 | 0.1266 | 0.7085 | 0.7009 | 0.0077 | 1.5429 | 0.1340 |
| 4096 | 1 | 1 | 0.0008 | 0.0483 | 0.0107 | 0.0107 | 0.0073 | 0.0218 | 0.0521 |
| 4096 | 128 | 1 | 0.0007 | 0.0480 | 0.3791 | 0.0112 | 0.0073 | 0.3810 | 0.0528 |
| 4096 | 128 | 128 | 0.0673 | 0.0803 | 0.3609 | 0.3774 | 0.0075 | 0.8287 | 0.0870 |
| 4096 | 512 | 512 | 0.5325 | 0.1835 | 1.3068 | 1.3234 | 0.0080 | 3.1231 | 0.1936 |

PPで必要な128行から1行の選択は、転送込みでは大幅に短縮する。TGの1行から1行ではCPU計算自体が極めて短く、XPU launch/status/wait費用が勝り、単体の総費用は増える。大規模な非同期化やunsafe fence削除は採用していない。

## 通常PP512/TG128

Single PD0、threads6、batch/ubatch512/128、F16 KV、同一binary/library。標準llama-bench -p512 -n128 -r1、observer/profileなし。要求contextは標準benchがPP512/空KV TG128で設定する値。GET_ROWS_F32=0,1,1,0のABBA。Tinyの1標本に大きな変動があったため1,0を各1回だけ追加した。外れた標本も除外していない。

| モデル | phase | n OFF/ON | OFF平均±SD token/s | ON平均±SD token/s | 中央値 OFF / ON | 平均変化 |
|---|---|---:|---:|---:|---:|---:|
| Llama 2 7B | PP | 2/2 | 80.224 ±0.016 | 80.232 ±0.075 | 80.224 / 80.232 | 0.01% |
| Llama 2 7B | TG | 2/2 | 8.534 ±0.010 | 8.516 ±0.013 | 8.534 / 8.516 | -0.21% |
| TinyLlama | PP | 3/3 | 318.822 ±2.275 | 319.903 ±2.898 | 317.846 / 318.901 | 0.34% |
| TinyLlama | TG | 3/3 | 28.733 ±0.015 | 28.352 ±0.657 | 28.738 / 28.715 | -1.33% |

Tiny ON TGには27.593 token/sの1標本があり、他のONは28.747/28.715。全標本平均では約1.33%低下、中央値では約0.08%低下となる。原因をkernelだけへ帰属できず、少数標本で明確な高速化とも継続的な1.33%回帰とも断定しない。7BもTG約0.21%低下、PPほぼ同等。追加の網羅的benchは行わず、**未設定/0はCPU、明示1のみXPUという既定OFFを維持する。**

## 転送とgraph

別実行のPP128からTG4、context1024の同期付き診断。TGは実decode3回の平均を表示する。copy API時間はinclusiveの診断値であり純粋なPCIe時間ではない。通常token/sへ混ぜず、他の時間へ加算しない。

| 指標 | 7B OFF -> ON | Tiny OFF -> ON |
|---|---:|---:|
| CPU F32 GET_ROWS（PP / TG3回） | 2/6 -> 0/0 | 2/6 -> 0/0 |
| XPU F32 GET_ROWS（PP / TG3回） | 0/0 -> 2/6 | 0/0 -> 2/6 |
| 対象ActivationのPP D2H本体 | 4,194,304 -> 0 B | 2,097,152 -> 0 B |
| 対象選択結果のPP H2D本体 | 32,768 -> 0 B | 16,384 -> 0 B |
| 全PP D2H | 4,421,376 -> 227,584 B | 2,293,168 -> 196,528 B |
| 全PP H2D | 223,772 -> 196,896 B | 207,580 -> 197,088 B |
| 全TG D2H/token | 162,304 -> 130,048 B | 145,344 -> 129,472 B |
| 全TG H2D/token | 66,288 -> 33,524 B | 55,685.3 -> 39,305.3 B |
| PP D2H API時間 | 2.637 -> 1.946 ms | 1.733 -> 1.276 ms |
| PP H2D API時間 | 0.832 -> 0.821 ms | 1.335 -> 1.403 ms |
| TG D2H API時間/token | 1.481 -> 1.514 ms | 1.028 -> 1.028 ms |
| TG H2D API時間/token | 0.529 -> 0.537 ms | 1.107 -> 1.108 ms |
| reserve graph分割 | 5 -> 3 | 5 -> 3 |
| normal bench KUNLUN graph実行（PP、warmup込み） | 18 -> 16 | 18 -> 16 |
| 同TG（warmup1+TG128の129step） | 387 -> 258 | 387 -> 258 |
| Single PD P2P | 0 | 0 |

normal graph数はbackend既存counterであり、observerでnodeごとに分断した診断値ではない。TGのKUNLUN graph起動は3から2/stepに減った。Activation共有groupは7B PP506/TG8256、Tiny PP346/TG5676でOFF/ON不変だった。

statusは2呼出しで512 B D2Hが追加される。PPのH2D総量には初回code upload 5,888 Bとout_ids 4 Bが入り、定常の選択結果H2D削減量とは異なる。attn_out/l_outのtensor_getはON診断で消失した。最終logitsはsampling用にCPUへ渡すため、約128,000 B/tokenの必要なD2Hは残る。KV index検証や他kernelの小さいmetadata転送も残る。

残る3splitはdebugで「KUNLUN Embedding -> CPU mixed入力分岐（node_1を入力とするDUP/SET_ROWS）-> KUNLUN Transformer」を確認した。通常token入力ではmixed分岐は非選択であり、CPU GET_ROWSの継続fallbackではない。observerにDUP/SET_ROWSが現れること自体をactive計算の証拠と扱わない。前回のskipでnode_1の大きなcopyはないが、split計画と依存待ちは残す。今回それを消すためのscheduler改変はしなかった。

## メモリ・採否・次の候補

| PP128診断 | OFF | ON | 差 |
|---|---:|---:|---:|
| 7B API-tracked HBM peak | 4,526,471,680 B | 4,526,477,824 B | +6,144 B |
| Tiny API-tracked HBM peak | 781,706,496 B | 780,664,064 B | -1,042,432 B |
| 7B GPU compute buffer | 22.38 MiB | 22.38 MiB | 同じ |
| 7B CPU compute buffer | 10.25 MiB | 8.25 MiB | -2 MiB |
| Tiny GPU compute buffer | 17.62 MiB | 16.62 MiB | -1 MiB |
| Tiny CPU compute buffer | 5.25 MiB | 4.25 MiB | -1 MiB |

新codeのallocation 5,888 Bと再利用status 256 Bのみ追加。TinyはplannerのGPU bufferが1 MiB減り、上記差分と一致する。Activationの全体shadowを新たに保持せず、weight ABI・pack・upload回数やL3容量は変えない。HBM値はSDK allocation hookのpeakで、全driver内部確保を保証する値ではない。

F32 GET_ROWSは正しくXPUに集約でき、PPの大きなreadbackは解消したが、LLMの大部分は既存Q4/Attention等に費やされ、削減は全体比で小さい。TGでは微小なCPU処理をlaunch/検証/待ちへ置き換える費用もある。独立opt-inとして採用し、defaultは変更しない。次は隣接する2行選択のlaunch/status検証の共通化、同値なidentity選択を省略できる厳密なgraph条件、非選択mixed分岐をsplit計画から外す安全な設計が候補。今回これらを無検証で適用していない。

最大温度60℃、80℃ guard・停止ラッチ・他利用者確認を継承。最終両PD RUNNING/no error、errtask不変。driver/firmware/clock/OS変更やresetなし。モデル複製、大規模trace、corpus評価は行わず空き容量約36.5 GBを維持。

## 再現・Git

機能と文書を両repoの独立branchへcommitし、llama.cppのvendor/KunlunK200 gitlinkを対応するlibrary HEADへ合わせる。最終commit一覧はdelivery-summary.jsonに保存する。GitHub push、PR、main/master更新は行わない。

~~~sh
R=/home/unagidojyou/kunlun-k200
O=$R/results/getrows-f32-20261101
L=$R/worktrees/getrows-f32-20261101-lib
P=$R/worktrees/getrows-f32-20261101-llama
# 既存SDK環境 XRE/XDNN/XTDK_DIR を使う。source build例:
cmake -S "$L" -B "$R/build/getrows-f32-source-lib" \
  -DCMAKE_BUILD_TYPE=Release -DKUNLUN_BUILD_BENCHMARKS=OFF \
  -DKUNLUN_XRE="$XRE" -DKUNLUN_XDNN="$XDNN" -DKUNLUN_XTDK="$XTDK_DIR" \
  -DCMAKE_INSTALL_PREFIX="$O/install"
cmake --build "$R/build/getrows-f32-source-lib" -j4 --target kunlunk200
cmake --install "$R/build/getrows-f32-source-lib"
# 今回は既存kernelをsymlinkで再利用し、新F32 imageだけ上記compiler/stack guardでbuild。
# その場合は -DKUNLUN_PREBUILT_KERNELS="$O/kernels" を指定。
cmake -S "$P" -B "$R/build/getrows-f32-20261101-llama" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_KUNLUN=ON \
  -DKUNLUN_USE_SYSTEM_LIBRARY=ON -DCMAKE_PREFIX_PATH="$O/install" -DLLAMA_BUILD_TOOLS=ON
cmake --build "$R/build/getrows-f32-20261101-llama" -j4 \
  --target llama-bench kunlun-llm-smoke kunlun-get-rows-f32-test
export GGML_KUNLUN_COMMON_Q4=1 GGML_KUNLUN_GET_ROWS=1 GGML_KUNLUN_SWIGLU=1
export KUNLUN_LARGE_PP=0 KUNLUN_Q6_K_SDNN=1 KUNLUN_Q6_K_LEGACY=0
export GGML_KUNLUN_Q4_ACT_SHARE=1 KUNLUN_Q4_MAC_DMA=1 KUNLUN_Q4_NMAJOR=1 KUNLUN_Q4_GEMM_OWNERS=1
export GGML_KUNLUN_GET_ROWS_F32=1 # 未設定/0でCPU F32 GET_ROWS
# thermal guard/latch/両PD確認付きrunner経由の再現:
python3 "$O/bench.py" # F32だけABBA、normal bench
python3 "$O/smoke.py" # PP512 logits・greedy8・KV復元
~~~

実際のbenchmark引数はllama-bench -m MODEL -p512 -n128 -r1 -b512 -ub128 -t6 -ngl99 -sm none -dev KUNLUN0 -mg0 -ctk f16 -ctv f16 -fa on -o json。正確なargv/環境と最小限の結果はresults/logs/*.json、summary.jsonに残す。GET_ROWS_F32.mdにAPI・fallback・overrideを記載。
