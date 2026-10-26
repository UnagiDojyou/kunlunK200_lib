# 再現手順

既存実機 `/home/unagidojyou/kunlun-k200` を前提とする。0.7.0のbuildは`build/flash-lib`、`build/flash-llama`に保持した。今回のbuildと結果は別ディレクトリにある。モデル、SDK、SSH秘密鍵は成果物に含めない。

## ビルド

```sh
R=/home/unagidojyou/kunlun-k200
export XRE="$R/external/xre" XDNN="$R/external/xdnn" XTDK_DIR="$R/external/xtdk"
export LD_LIBRARY_PATH="$XRE/lib64:$XDNN/so:$XTDK_DIR/shlib"

cmake -S "$R/repos/KunlunK200" -B "$R/build/flash-online-lib" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$R/build/flash-online-lib" -j6

cmake -S "$R/repos/llama.cpp" -B "$R/build/flash-online-llama" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_KUNLUN=ON -DKUNLUN_USE_SYSTEM_LIBRARY=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_MTMD=OFF
cmake --build "$R/build/flash-online-llama" -j6 --target \
  kunlun-llm-smoke kunlun-quality-eval kunlun-attention-test kunlun-swiglu-test kunlun-backend-contract
```

source基準はlibrary `780fb69`、llama.cpp `736fad5`。新しい製品カーネルは`kunlun-flash-online-ordered`、既存`kunlun-flash-online`は未変更。`auto`/未設定は新経路、`legacy`と不明値は旧経路、`ordered`は新経路の明示alias。

## 単体・API比較

```sh
K="$R/build/flash-online-lib/lib/kunlunk200/kernels"
"$R/build/flash-online-lib/k200-flash-online-bench" 0 128 512 128 32 1 1 "$K"
"$R/build/flash-online-lib/k200-flash-online-bench" 1 128 1024 1 32 1 1 "$K"
KUNLUN_FLASH_ONLINE=auto KUNLUN_FLASH_BENCH_SAMPLE=1 \
  "$R/build/flash-online-lib/k200-flash-online-api-bench" 0 128 2048 128 8 32 12
```

単体の引数は`PD D block M H mask phase kernel-directory`。maskは0=なし、1=F16 causal、2=F32 causal、4=全mask、6=非有限/極値Score、7=全half bit pattern（M128/block512）。phaseは0=初期化、1=途中block、2=最大値100の不変状態、3=最終正規化。入力復元コピーは計時外、単体は20回の中央値と最小値、enqueue時間を出力する。L3 workspaceは16 MiB以内が必要。

API benchmarkは同一プロセスで指定経路とlegacyを測り、全要素ビット一致を確認する。CPU FP64参照だけは`SAMPLE=1`で4query行に制限する。`KUNLUN_FLASH_FULL_Q_REF=1`は元のFP32 QをCPU参照に用い、既存Q cast誤差も含める。`KUNLUN_FLASH_BENCH_MUTATE=1`はsame-pointer Q/K/V更新と入力不変性を確認する。

## 通常のLLM性能測定

```sh
O="$R/results/flash-online-20261026"
export OMP_WAIT_POLICY=PASSIVE
export GGML_KUNLUN_COMMON_Q4=1 KUNLUN_BROADCAST_VECTOR=1
export GGML_KUNLUN_ROPE=1 GGML_KUNLUN_SWIGLU=0
export GGML_KUNLUN_ATTENTION=1 KUNLUN_ATTENTION_MATMUL=auto
export KUNLUN_ATTENTION_SOFTMAX=fast KUNLUN_ATTENTION_PREP=auto
export KUNLUN_FLASH_KV=1 GGML_KUNLUN_FLASH=1 K200_FLASH=1
export K200_CTX=4096 K200_BATCH=128 K200_UBATCH=128
export K200_INPUT_IDS="$O/inputs/prompt2048.ids" K200_TEACHER_FILE="$O/inputs/teacher64.ids"
export K200_PP_LENGTHS=128,512,2048 K200_WARMUP=1 K200_REPEATS=2 K200_IGNORE_EOS=1
unset KUNLUN_FLASH_TRACE K200_TRACE_OPS K200_CHECK_STATE LD_PRELOAD
unset KUNLUN_FLASH_BLOCK KUNLUN_FLASH_QUERY_TILE KUNLUN_FLASH_PV KUNLUN_FLASH_TG_GROUP

KUNLUN_FLASH_ONLINE=legacy "$R/build/flash-online-llama/bin/kunlun-llm-smoke" \
  "$R/models/tinyllama-q4_0.gguf" none 16 "$O/repro-legacy.bin" 0
KUNLUN_FLASH_ONLINE=auto "$R/build/flash-online-llama/bin/kunlun-llm-smoke" \
  "$R/models/tinyllama-q4_0.gguf" none 16 "$O/repro-auto.bin" 0
```

7Bは`llama2-7b-q4_0.gguf`、PD1は末尾の0を1へ変更する。`none`を`layer`/`row`にするとSplit構成を確認できる。warmupと測定の区別は`K200PERF trial=...`を用いる。PPはtokens/pp_ms、TGはtg_steps/tg_msから計算する。通常性能には以下の診断設定を混ぜない。

## 保存スクリプトと診断

結果ディレクトリの`scripts/`または本成果物の`client-scripts/`に全スクリプトを保存した。実機で実行する評価scriptは結果ディレクトリ直下にも配置した。

- `units-final.py`: 102件の代表/端数/stride/broadcast/mask/KV更新試験。
- `perf-final.py`: 両モデル・両PDの通常性能。teacher-forcedの同一token列を使用。
- `ordered-corpus.py`: 1024 token corpusを0.7.0保存logitsと比較。参照は前回の`results/flash-attention-20261026`。
- `compat-final.py`: Single/Layer/Row、生成、state restore、既存演算回帰。末尾に補助診断と最終確認も実行する。
- `profile-build.py`/`profile-run.py`: `get_clock()`計測を入れた別カーネル。製品カーネルには計測を入れていない。
- `profile-split-build.py`/`profile-extra-run.py`: expとsumを分離する補助診断、最大値不変時の診断。追加LM loadとclockがあるため通常APIの内訳として加算しない。
- `collect-metrics.py`/`make-report.py`: 生ログからCSV/JSONと日本語レポートを生成。
- `final-checks.py`: 配布用bench build、元FP32 QのFP64比較、dispatch fallback、50 kernel SHA、最終デバイス状態。

`runner.py`は既存thermal guard、他プロセス確認、デバイスstate監視を利用する。試験失敗時には後続試験を止める。ホスト設定の誤りによる失敗とデバイス異常を区別してログを調べる。

`setup`、`install-candidates`、`reuse-candidate`、`ordered-candidate`、`adopt-ordered`、`stop-exploratory`などは開発履歴であり、現在のsourceに順不同で再実行しない。採用済みsourceの再現には上のbuild手順と評価scriptを使う。

再実行はモデル/SDKがある実機で、他の負荷がないことを確認した上で行う。通常性能と診断を同時実行しない。raw logitsは大容量のため実機結果ディレクトリに保持し、成果物には比較統計とSHAを含めた。
