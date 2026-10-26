# 0.7.4 KV最大値キャッシュの再現

実機workspaceは `/home/unagidojyou/kunlun-k200`。ソースは `repos/KunlunK200` と `repos/llama.cpp`、結果は `results/kv-max-20261026` に保存している。旧0.7.3 buildは `build/flash-decode-lib` / `build/flash-decode-llama`。

## ビルド

```sh
cd /home/unagidojyou/kunlun-k200
export XRE="$PWD/external/xre"
export XDNN="$PWD/external/xdnn"
export XTDK_DIR="$PWD/external/xtdk"
export LD_LIBRARY_PATH="$XRE/lib64:$XDNN/so:$XTDK_DIR/shlib"

cmake -S repos/KunlunK200 -B build/kv-max-lib -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/kv-max-lib -j4 --target k200-kv-max-bench k200-kv-max-contract k200-kv-max-alias

cmake -S repos/llama.cpp -B build/kv-max-llama -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_KUNLUN=ON -DKUNLUN_USE_SYSTEM_LIBRARY=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_MTMD=OFF
cmake --build build/kv-max-llama -j4 --target kunlun-llm-smoke kunlun-quality-eval
```

新backendはライブラリ0.7.4以上が必要。system library構成は `-DKUNLUN_USE_SYSTEM_LIBRARY=ON -DCMAKE_PREFIX_PATH=<0.7.4 install prefix>` でビルドできる。実機の `results/kv-max-20261026/install-074` でinstall/runtime kernel読込とsystem backend buildを試験した。SDK/モデルは成果物に同梱していない。

artifactから再構築する場合、`source/library` はライブラリの全ソース。llama.cppは保存されたbase commit `90d64614e749240d271dbef2863d588157ceb825` の別checkoutへ `patches/llama.patch` を適用する。system library構成では `git apply --exclude=vendor/KunlunK200` とし、別途このライブラリをbuild/installできる。既存の変更があるcheckoutへ直接上書きしない。

## 設定

| 設定 | 動作 |
|---|---|
| `KUNLUN_FLASH_KV_MAX=auto`（既定） | 所有・更新・finite/zero条件が確認できた場合だけ再利用 |
| `legacy` | 毎APIのK/V SDK findmax |
| `verify` | hitとSDK maxをbit比較。不明データをCLで再検証するため遅い |
| `KUNLUN_FLASH_KV_MAX_MIN_S` | 256/512/1024/2048/4096/8192/16384を指定可能。閾値を下げる用途は比較試験用 |

既定autoはfeature数 `D*HK >= 1024` でS512以上、256以上1024未満でS8192以上。256未満は自動有効化しない。capacityが閾値未満ならwriterも従来経路。未検証のhost書き込み・restore、alias、非対応view等はSDKへ戻る。

比較用設定 `KUNLUN_FLASH_KV_MAX_REBUILD=rows`, `KUNLUN_FLASH_KV_MAX_REDUCE=linear`, `KUNLUN_FLASH_KV_MAX_PREFILL=fused`, `KUNLUN_FLASH_KV_MAX_WRITER=lazy` は通常計測ではunsetにする。

## 単体試験

他プロセスと温度を既存の `check_users.py` / thermal guardで確認してから実行する。保存したclient scriptsはrunnerを通している。

```sh
# PD, D, S, HK。query=1、H=32。
KUNLUN_FLASH_DECODING=off build/kv-max-lib/k200-kv-max-bench 0 128 4096 32
KUNLUN_FLASH_DECODING=on  build/kv-max-lib/k200-kv-max-bench 1 64 8192 4

# PD,D,S,M,HK,H,pattern,batch。pattern12=F16 mask。
KUNLUN_FLASH_KV_MAX=verify KUNLUN_FLASH_DECODING=off \
  build/kv-max-lib/k200-kv-max-contract 0 64 512 128 4 32 12 1
KUNLUN_FLASH_DECODING=on build/kv-max-lib/k200-kv-max-alias 0
```

`KV_PERF` はKV不変のsteady API中央値。`KV_APPEND` は2 writerとAPIを含む16 appendの中央値で、入力uploadだけをtimer外に置く。`KV_CASE`/`KV_COMPARE`/`FLASH_MUTATE` は旧SDNN出力とのbit比較。通常の単体測定ではTRACE/PROFILE/verify環境を外す（bench自身の精度確認部分はverifyを使用する）。

## 通常LLM性能

```sh
export OMP_WAIT_POLICY=PASSIVE
export GGML_KUNLUN_COMMON_Q4=1 KUNLUN_BROADCAST_VECTOR=1
export GGML_KUNLUN_ROPE=1 GGML_KUNLUN_SWIGLU=0
export GGML_KUNLUN_ATTENTION=1 KUNLUN_ATTENTION_MATMUL=auto
export KUNLUN_ATTENTION_SOFTMAX=fast KUNLUN_ATTENTION_PREP=auto
export KUNLUN_FLASH_KV=1 GGML_KUNLUN_FLASH=1 K200_FLASH=1
export KUNLUN_FLASH_KQ=auto KUNLUN_FLASH_ONLINE=auto KUNLUN_FLASH_DECODING=auto
export K200_CTX=4096 K200_BATCH=128 K200_UBATCH=128
export K200_INPUT_IDS="$PWD/results/kv-max-20261026/inputs/prompt2048.ids"
export K200_TEACHER_FILE="$PWD/results/kv-max-20261026/inputs/teacher64.ids"
export K200_PP_LENGTHS=128,512,1024,2048 K200_WARMUP=0 K200_REPEATS=1 K200_IGNORE_EOS=1

KUNLUN_FLASH_KV_MAX=legacy build/flash-decode-llama/bin/kunlun-llm-smoke \
  models/llama2-7b-q4_0.gguf none 32 /tmp/k200-old.bin 0
KUNLUN_FLASH_KV_MAX=auto build/kv-max-llama/bin/kunlun-llm-smoke \
  models/llama2-7b-q4_0.gguf none 32 /tmp/k200-new.bin 0
```

PD1は最後の引数を1にする。Layer/Rowは`none`を`layer`/`row`にしてmain PD=0。Tinyはモデルを`tinyllama-q4_0.gguf`へ変更。prefix4096はctx4608・prompt4096.ids・PP_LENGTHS4096・64 positions、Tiny8192はctx8704・prompt8192.ids・teacher128.ids・128 positionsを使用した。

元のsmokeツールはfixture全長をcontextと照合してからPP_LENGTHSで切るため、入力file全長を入れられるcontextを指定する。

## 品質・state

`kunlun-quality-eval model none ids context dump reference` でPPL/logits/KLを測定できる。`K200_QUALITY_BATCH=128`/`1`を指定する。1024 IDs/ctx256/score1020、および512 IDs/ctx512/score511を使用した。Tinyのcache-active品質試験だけは `MIN_S=256` を強制し、通常の短context性能とは区別した。

smokeの `K200_CHECK_STATE=1 K200_CHECK_KV_MUTATIONS=1` はrestore・overwrite・rollback・clear/reuseを検証し、通常性能測定とは分離する。`K200_CHECK_KV_SHIFT=1` は既存のF16 in-place RoPE非対応を再現する診断用であり、成功するテストとして扱わない。

`kunlun-kv-max-bench`（llama側）はrestoreを伴うA/B診断。autoは未検証のrestore後にSDK fallbackするため、通常append性能の主要測定には使用しない。通常推論には上記smokeを使う。

## 保存物

- `REPORT.ja.md`: 採用・不採用、性能表、品質、制約。
- `logs/*.txt, *.json`: rawログ、環境、argv、returncode、device state、実行時間。
- `data/*.json`: grid、段階別診断、source/kernel監査、集計。
- `inputs/`: 同じ入力を再現するtoken IDs。
- `source/`, `patches/`, `COMMITS.json`: 動作ソースと変更点。
- `runtime-0.7.4.tar.gz`: 自作ライブラリ・header・kernelのinstall tree。

大量のlogits dumpは実機の結果directoryに保持し、local artifactにはSHA256 manifestを付けた。SDKやモデル・秘密鍵は同梱しない。
