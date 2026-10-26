# KQ最適化の再現手順

実機 `/home/unagidojyou/kunlun-k200` のSDK・モデルを使用する。モデル、SDK、SSH秘密鍵は成果物に含めない。0.7.1の既存buildは`build/flash-online-lib`/`flash-online-llama`に保持した。新sourceは`repos/KunlunK200`、新buildは`build/flash-kq-lib`/`flash-kq-llama`。

## build

```sh
R=/home/unagidojyou/kunlun-k200
export XRE="$R/external/xre" XDNN="$R/external/xdnn" XTDK_DIR="$R/external/xtdk"
export LD_LIBRARY_PATH="$XRE/lib64:$XDNN/so:$XTDK_DIR/shlib"
cmake -S "$R/repos/KunlunK200" -B "$R/build/flash-kq-lib" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$R/build/flash-kq-lib" -j6
cmake -S "$R/repos/llama.cpp" -B "$R/build/flash-kq-llama" -G Ninja \
 -DCMAKE_BUILD_TYPE=Release -DGGML_KUNLUN=ON -DKUNLUN_USE_SYSTEM_LIBRARY=OFF \
 -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=OFF \
 -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_MTMD=OFF
cmake --build "$R/build/flash-kq-llama" -j6 --target kunlun-llm-smoke kunlun-quality-eval \
 kunlun-attention-test kunlun-swiglu-test kunlun-backend-contract
```

`KUNLUN_FLASH_KQ=auto`（未設定も同じ）が新方式＋短block fallback、`legacy`が旧KQ、`reuse1024`が新KQの強制指定。不明値は旧KQ。FlashAttention自体は既存通りopt-inである。`KUNLUN_FLASH_ONLINE=auto`を両方式で共通にする。

## KQ単体・Flash API

```sh
K="$R/build/flash-kq-lib/lib/kunlunk200/kernels"
# PD D S_source M HK Hq kernel-directory [pattern batch query-tile [suffix]]
"$R/build/flash-kq-lib/k200-flash-kq-bench" 0 128 2048 128 32 32 "$K"
"$R/build/flash-kq-lib/k200-flash-kq-bench" 1 64 2048 1 4 32 "$K"
# PD D S M HK Hq pattern [batch]
KUNLUN_FLASH_KQ=auto KUNLUN_FLASH_ONLINE=auto KUNLUN_FLASH_BENCH_SAMPLE=1 \
 "$R/build/flash-kq-lib/k200-flash-kq-api-bench" 0 128 2048 128 32 32 12
```

単体はS_sourceの末尾にあるPP最大512/TG最大1024キーの1 blockを測る。Q/K/range/peak準備は計時外、1 warmup＋12回の上側中央値。Kは4部分最大値の最後が最大となるpeak配列で確認し、全出力を旧KQとbit比較する。CPU FP64はPP4query/全headを参照。TG GQAのScoreは正規化Qを使ったCL復元前の値。stride、head別scale、batch等の試験は`extra-units.py`にある。

API benchmarkは全Sのcast/range/findmaxから最終waitまでを測定し、同一プロセス内でlegacyと比較する。CPU参照のみ`SAMPLE=1`で4queryに絞り、旧出力との差は全要素を確認する。`KUNLUN_FLASH_FULL_Q_REF=1`は元FP32 Qのcast誤差も含める。`KUNLUN_FLASH_BENCH_MUTATE=1`はQ/K/V same-pointer更新とread-only不変性を確認する。

## 通常推論

```sh
O="$R/results/flash-kq-20261026"
export OMP_WAIT_POLICY=PASSIVE
export GGML_KUNLUN_COMMON_Q4=1 KUNLUN_BROADCAST_VECTOR=1
export GGML_KUNLUN_ROPE=1 GGML_KUNLUN_SWIGLU=0
export GGML_KUNLUN_ATTENTION=1 KUNLUN_ATTENTION_MATMUL=auto
export KUNLUN_ATTENTION_SOFTMAX=fast KUNLUN_ATTENTION_PREP=auto
export GGML_KUNLUN_FLASH=1 K200_FLASH=1 KUNLUN_FLASH_KV=1 KUNLUN_FLASH_ONLINE=auto
export K200_CTX=4096 K200_BATCH=128 K200_UBATCH=128 K200_IGNORE_EOS=1
export K200_INPUT_IDS="$O/inputs/prompt2048.ids" K200_TEACHER_FILE="$O/inputs/teacher64.ids"
export K200_PP_LENGTHS=128,512,2048 K200_WARMUP=1 K200_REPEATS=2
unset KUNLUN_FLASH_TRACE K200_TRACE_OPS K200_CHECK_STATE LD_PRELOAD
unset KUNLUN_FLASH_BLOCK KUNLUN_FLASH_QUERY_TILE KUNLUN_FLASH_PV KUNLUN_FLASH_TG_GROUP
KUNLUN_FLASH_KQ=legacy "$R/build/flash-kq-llama/bin/kunlun-llm-smoke" \
 "$R/models/tinyllama-q4_0.gguf" none 16 "$O/repro-legacy.bin" 0
KUNLUN_FLASH_KQ=auto "$R/build/flash-kq-llama/bin/kunlun-llm-smoke" \
 "$R/models/tinyllama-q4_0.gguf" none 16 "$O/repro-auto.bin" 0
```

7Bは`llama2-7b-q4_0.gguf`、PD1は末尾0を1へ、Splitは`none`を`layer`/`row`へ変更する。通常CLIでは`-fa on`を指定する（評価toolは`K200_FLASH=1`）。PPはtokens/pp_ms、TGはtg_steps/tg_msから計算する。短PPは差が小さいので各反復と測定順を保存し、初回と追試を混ぜて最良値だけを採用しない。

## 保存した評価スクリプト

`results/flash-kq-20261026/scripts/`と成果物の`client-scripts/`に保存した。主要評価scriptとrunner/helperは結果ディレクトリ直下にも配置する。

- `units-final.py`: API代表/互換性102件。
- `extra-units.py`: 単体grid・query tile sweep・GQA/stride/外れ値と段階診断。
- `profile-build.py`: legacy/reuse32/reuse1024から別の計時カーネルを生成。実コードにclockを入れない。
- `small-shapes.py`: 再利用がない短sequenceの比較。
- `corpus.py`: 0.7.1保存logitsとの1024 token corpus比較。
- `perf-final.py`: 通常の両モデル・両PD・PP/TG。
- `repeat-tiny.py`/`repeat-7b-short.py`: 順序反転・3反復の追試。
- `compat-final.py`: Single/Layer/Row、短/長context生成、state restore、既存回帰。
- `final-checks.py`: auto/fallback実経路、FP32 Q込みFP64比較、特殊条件、最終auto corpusと状態。
- `collect.py`/`make-report.py`: CSV/JSONと日本語レポート生成。

`runner.py`は既存thermal guard、他プロセス確認、device state監視を使い、失敗すると後続試験を止める。通常測定と診断を同時に走らせない。`setup/install/add-wide/adopt/fix-build`等のmutatorは開発履歴であり、現sourceへ順不同で再実行しない。

診断t0..t9はQ DMA、Q DS、K DMA、K DS、MAC、EW、RS、DMAOUT、その他制御、合計のtick。t10..t15はQ DMA要求、K DMA要求、Q DS batch、K DS batch、MAC、DMAOUTの実行数。900 MHz換算を用い、fence込み・profiling overheadありと明示する。単体診断のsuffixは`-profile-legacy`等であり、normalのtoken/sとは別。

比較候補32/128/256は製品にbuild/installしない。`bench/experiments/flash-kq/`のsourceと`experimental-kernels/`のbinを保存した。必要なら既存`cmake/compile-native.sh`で別出力先へcompileし、その出力先にlegacy bin/entryも置いて、単体benchのsuffix引数に`-reuse32`等を指定する。

Git HEADだけでは0.7.1の未コミット成果を再現できない。保存snapshotと`changes-since-0.7.1.patch`、または成果物の完成sourceを使う。raw logitsは大容量のため実機結果ディレクトリへ保持し、成果物には統計・SHA・ログ・入力token列を含める。
