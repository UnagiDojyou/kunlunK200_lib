# Gate/Up Pair追加opt-in・再現手順

基準はgate-up-auto-20261103と完成済みPreaddr/DMA_CONST。既定値は変更しない。

| flag | 未設定/0/不正値 | 1 |
| --- | --- | --- |
| KUNLUN_Q4_PAIR_SUM_FAST | 既存pair sum | ordered K32、16-block読込のPair専用CL sum |
| KUNLUN_Q4_QUANT_EXACT | 既存Max/Main/Residual | F32 M1、K=2048/4096/5632/11008の整数max/word初期化 |

Pair本体はGGML_KUNLUN_Q4_GATE_UP_PAIR未設定/auto/1で既存条件に従い、0で従来GEMV。対応はK=4096,N=11008とK=2048,N=5632のM1ペア、ABI-v1、同じActivation・PD、aliasなし、有効LeaseとL3容量が必要。非対応shape、graph分割、alias等は既存fallback判定を維持する。量子化opt-inはPair以外の対応M1 GEMVにも適用され、F16・他K・GEMMは旧経路。

必要な新imageはkunlun-common-sum-pair-fast、kunlun-common-max-exact、kunlun-common-main-exact、kunlun-common-residual-exact（各bin/entry）。古いimageは削除しない。PreaddrとDMA_CONSTも従来どおり独立opt-in、既定OFF。SUM_FASTとSWIGLU_ABSMAXは今回0固定。新旧の実行条件はtheme5/profiles.jsonのA/B/Cを使う。

## 保存状態の復元

final-state.jsonのsource.final.library/llamaのbase_commitと累積patchを使う。既存worktreeを上書きせず、一意な名前の新worktreeを作り、final-library.patch / final-llama.patchを適用する。sourceファイルSHAを保存manifestと照合する。保存HEADだけで実装を識別しない。

theme5/build-commands.jsonは実際に使った絶対パスの記録である。そのまま既存build/installへ再実行せず、worktree/build/private installの3種類のパスを新しい再現用パスへ置換して実行する。CMakeのRelease、XRE/XDNN/XTDK、コンパイル条件は同じにする。KUNLUN_PREBUILT_KERNELSは保存済みtheme5/prebuiltを読取り使用できる。imageをsourceから再生成する場合はcmake/compile-native.shとcompile-commandsを参照し、SDKは変更しない。使用imageと同一になることはtheme5/source-rebuild.jsonで確認済み。

source最小差分だけを確認する場合は、保護されたscale-resume版へ書き込まず、その累積patchから作った別worktreeへproduction-only.patchを適用する。final-sourceにはdirty sourceのsnapshot、runtimeには使用binaryのコピー、ldd-*.txtには実際に解決したDSOを保存する。

## CPUと安全監視

このHostではCPU2をmain threadにのみ指定する。通常推論は自動固定しない。別Hostでは /sys/devices/system/cpu/cpu*/topology、/proc/interrupts、/proc/irq/*/effective_affinity_list、許可CPU maskから選択する。worker/IRQ/governorは変更しない。標準benchmarkのLLAMA_BENCH_MAIN_CPU=2を使用し、before/applied/measured_end/restoredログを確認する。

通常の再測定も既存thermal guardとepoch-aware runnerから行う。runner.pyのconfigure先は新しいresultsへ変更し、check_users.py、recovery-epoch.json、raw旧errtaskのSHAを検証する。過去のerrtaskを新しい異常として無条件に承認しない。reset_countやraw SHAが変わった場合はfail-stopする。監視を外して直接実機benchmarkを起動しない。

## 通常ベンチマークの内容

環境はtheme5/profiles.json、各runの完全引数はlogs/final-*.jsonに保存。主な引数は次のとおり。

```bash
llama-bench -m EXISTING_MODEL.gguf -p 512 -n 128 -r 1 -b 512 -ub 128 -t 6 -ngl 99 -sm none -dev KUNLUN0 -mg 0 -ctk f16 -ctv f16 -fa on -o json
```

連続実行は-p 0 -n 0 -pg 512,128とLLAMA_BENCH_SPLIT_TIMING=1を使用する。独立PP/TGと同じ値として集計しない。通常測定ではLD_PRELOAD audit、K200_STEP_PROBE_DIR、SDK_PROFILE、device-clock traceを入れない。

各条件3独立processの順序巡回を使い、process中央値/平均/SD/CVとrawを残す。同一processの31micro反復を独立nと扱わない。micro用measure.pyは既存ログ保護・完了markerがあるため、追試には別resultsを作る。

theme1/failed-abi-transplantのimageは内部ABI不整合で不採用。再実行しない。正しいGroup Ablationはtheme1_v2.pyとtheme1/imagesに保存する。新しいDMA方式、設定保持の仮定、fence/barrier削除、1-core量子化統合は今回採用していない。
