# Flash-Decoding 0.7.3 再現手順

実機の作業rootは `/home/unagidojyou/kunlun-k200`。SDK・モデルは既存のものを使用する。SSHの秘密鍵は成果物へ含めていない。

## 比較基準

- library 0.7.2: `d8bd87f56ae95872287c4f4a030b7ac7e68dd7b9`
- llama.cpp: `8d62617cb1d267d9283d5fcffe8e66c57820dfcc`
- 凍結ビルド: `build/flash-kq-lib`、`build/flash-kq-llama`
- 新ビルド: `build/flash-decode-lib`、`build/flash-decode-llama`
- 実験結果: `results/flash-decoding-20261026`

配布のbaseline-source archiveとpatch、またはsource archiveを使用する。llama.cppのvendor/KunlunK200には同じ配布のlibrary sourceを配置する。libraryとshaderは必ず同じビルドの組を使用する。private metadataは192 byte、旧FAの公開契約は変更していない。

## ビルド

```sh
export R=/home/unagidojyou/kunlun-k200
export XRE="$R/external/xre" XDNN="$R/external/xdnn" XTDK_DIR="$R/external/xtdk"
export LD_LIBRARY_PATH="$XRE/lib64:$XDNN/so:$XTDK_DIR/shlib"
cmake -S "$R/repos/KunlunK200" -B "$R/build/flash-decode-lib" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$R/build/flash-decode-lib" -j4 --target k200-flash-decode-bench k200-flash-contract k200-attention-contract k200-common-q4-layout
cmake -S "$R/repos/llama.cpp" -B "$R/build/flash-decode-llama" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_KUNLUN=ON -DKUNLUN_USE_SYSTEM_LIBRARY=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_MTMD=OFF
cmake --build "$R/build/flash-decode-llama" -j4 --target kunlun-llm-smoke kunlun-quality-eval kunlun-flash-decoding-bench kunlun-attention-test kunlun-swiglu-test kunlun-backend-contract
```

このcheckout/configurationにはllama-cli targetがないため、評価には上記の専用ツールを使う。SDK本体、XTDK header、モデルGGUFは別途必要。

## 単体API、数値、fallback

引数は `PD D S M HK Hq pattern [batch]`。初回を除く4回の中央値で、CPU FP64・旧FAと出力を比較し、入力不変性とguardを確認する。

```sh
KUNLUN_FLASH_DECODING=on KUNLUN_FLASH_DECODING_CHUNKS=4 "$R/build/flash-decode-lib/k200-flash-decode-bench" 0 128 4096 1 1 32 12 1
KUNLUN_FLASH_DECODING=on KUNLUN_FLASH_DECODING_FIXED_KEYS=1024 KUNLUN_FLASH_DECODING_WORK=head "$R/build/flash-decode-lib/k200-flash-decode-bench" 1 64 4352 1 4 32 29 1
KUNLUN_FLASH_DECODING=auto "$R/build/flash-decode-lib/k200-flash-decode-bench" 0 64 4352 1 4 32 29 1
KUNLUN_FLASH_DECODING=auto "$R/build/flash-decode-lib/k200-flash-decode-bench" 0 128 4352 1 32 32 29 1
```

pattern29はinterleaved compact KVとF16 mask（末尾255 keyをmask）。0と11は同じ論理入力・F32 maskで、KVの物理配置だけが異なる。12はF16 mask。3/11はstride/view、6は全mask、18はbatch broadcast、24/25は非zero mask＋ALiBi、27/28は一部chunkの全mask。

`CHUNKS=1..64`、`CHUNK=256/512/1024`はbalanced modeの調整。`FIXED_KEYS=32/64/128/256/512/1024`は実chunk長を固定してchunk数を決定し、前二者に優先する。`WORK=chunk/head/auto`、`LAYOUT=head/chunk`も比較可能。詳細はlibrary docs/flash-decoding.md。

`COMPARE_WORK=1`、`COMPARE_LAYOUT=1`、`COMPARE_FIXED=1`（いずれもKUNLUN_FLASH_DECODING_ prefix）を指定した試験ではold_msはその比較対象であり、0.7.2ではない。通常比較はこれらをunsetする。

## 通常LLM性能

```sh
export O="$R/results/flash-decoding-20261026"
export OMP_WAIT_POLICY=PASSIVE GGML_KUNLUN_COMMON_Q4=1 KUNLUN_BROADCAST_VECTOR=1
export GGML_KUNLUN_ROPE=1 GGML_KUNLUN_SWIGLU=0 GGML_KUNLUN_ATTENTION=1
export KUNLUN_ATTENTION_MATMUL=auto KUNLUN_ATTENTION_SOFTMAX=fast KUNLUN_ATTENTION_PREP=auto
export KUNLUN_FLASH_KV=1 GGML_KUNLUN_FLASH=1 K200_FLASH=1
export KUNLUN_FLASH_KQ=auto KUNLUN_FLASH_ONLINE=auto
export K200_CTX=4608 K200_BATCH=128 K200_UBATCH=128
export K200_INPUT_IDS="$O/inputs/prompt4096.ids" K200_TEACHER_FILE="$O/inputs/teacher64.ids"
export K200_PP_LENGTHS=4096 K200_REPEATS=2 K200_WARMUP=1 K200_IGNORE_EOS=1
unset LD_PRELOAD KUNLUN_FLASH_TRACE KUNLUN_FLASH_DECODING_PROFILE K200_TRACE_OPS K200_CHECK_STATE
unset KUNLUN_FLASH_DECODING_CHUNKS KUNLUN_FLASH_DECODING_CHUNK KUNLUN_FLASH_DECODING_FIXED_KEYS KUNLUN_FLASH_DECODING_WORK
KUNLUN_FLASH_DECODING=off "$R/build/flash-decode-llama/bin/kunlun-llm-smoke" "$R/models/tinyllama-q4_0.gguf" none 64 "$O/repro-off.bin" 0
KUNLUN_FLASH_DECODING=auto "$R/build/flash-decode-llama/bin/kunlun-llm-smoke" "$R/models/tinyllama-q4_0.gguf" none 64 "$O/repro-auto.bin" 0
```

PD1は最後の引数を1へ変更。Splitはnoneをlayer/rowへ変更しdevice引数は0。2kまでのmatrixはprompt2048.ids、ctx4096、PP_LENGTHS=128,512,1024,2048。teacher固定で63回decodeする。prompt4096は2048-token入力を2回連結したfixture。

## 交互TG比較

同じ環境で、`K200_FD_MODES=off,auto K200_REPEATS=2` を設定し、実行ファイルをkunlun-flash-decoding-benchへ変更する。共通PPを1回実行し、各trialの前にclear/state restoreする。復元・dump・checksumはTGタイマー外。各mode内のlogits checksumが反復で一致しなければ失敗する。

探索用modeは `off,on,on:5,head:5,fixed,fixhead`。on:5はbalanced5、head:5は同じ境界のhead affinity、fixed/fixheadは固定長1024のchunk/ head割当。採用後のonは既知shapeを自動調整するので、探索時のbalanced5を再現するときはCHUNKS=5とWORK=chunkを明示する。autoを正確に再現する場合は調整変数をunsetする。

## 品質・監査

標準コーパスはcorpus1024.ids、ctx256、1020 scored positions。K200_QUALITY_BATCH=1でTGを実行し、offとonを同条件で比較。強制CHUNKS=4とFIXED_KEYS=64はctx256で同じ分割となり、出力のビット一致を確認した。batch128はPPであり、旧版と完全一致する。

```sh
K200_QUALITY_BATCH=1 KUNLUN_FLASH_DECODING=off "$R/build/flash-decode-llama/bin/kunlun-quality-eval" "$R/models/tinyllama-q4_0.gguf" none "$O/inputs/corpus1024.ids" 256 "$O/ppl-off.bin" - -
K200_QUALITY_BATCH=1 KUNLUN_FLASH_DECODING=on KUNLUN_FLASH_DECODING_FIXED_KEYS=64 "$R/build/flash-decode-llama/bin/kunlun-quality-eval" "$R/models/tinyllama-q4_0.gguf" none "$O/inputs/corpus1024.ids" 256 "$O/ppl-on.bin" "$O/ppl-off.bin" -
```

`KUNLUN_FLASH_TRACE=1`、`KUNLUN_FLASH_DECODING_PROFILE=1`は追加同期を含む診断用。0を代入しても有効になるため通常測定ではunsetする。SDNN phaseはfence区間＋get_clock、CLはworkerごとの処理行数・経過clockを記録する。profiling時のFLASH_UNITを通常速度として使わない。

runner.py/check_users.pyは他プロセスのXPU使用・温度・device stateを確認し、失敗時はXPU_ABORTで後続を停止する。保存済み実験へ再実行するときは、incidentの解決を確認してから使う。過去のソース書換え用client scriptsは開発履歴であり、完成済みtreeへ一括で再実行しない。
