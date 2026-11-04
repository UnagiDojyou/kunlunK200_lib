# Activation量子化拡張・再現手順

基準はresults/remaining-four-20261104の最終高速profile C。新成果物はresults/activation-two-20261104。

## P8拡張selector

KUNLUN_Q4_QUANT_PACK_EXTは既定OFF。未設定/0/不正値では前回と同じK2048だけのP8。1でK5632/11008の追加対象を有効化、5632または11008でそのKだけを有効化する。QUANT_EXACT=1、QUANT_PACK=1、F32 M1が必要。K4096は今回対象外でQ2のまま。K2048の既存P8はEXTの値に依存しない。

拡張は既存kunlun-common-main-pack/residual-pack imageを共有するHost dispatchだけ。新しい通常imageやprogram slot、data HBM/L3、同期を追加しない。既存の丸め、clamp、独立scale/inverse除算、little-endian pack、zero tail、Lease/generation/consumerは維持する。

K2048のidle-core試作R/MRは性能が安定しないため不採用であり、通常backendへ導入していない。KUNLUN_Q4_QUANT_IDLEはprobe記録上の条件名だけで、通常公開opt-inではない。final installへidle imageを追加しない。

final/profiles.jsonのAが前回高速版、Dが今回の拡張。要求条件BはD、CはAと同じなので、独立標本を水増しせずA/Dのみ測る。共通でPreaddr/DMA_CONST/PAIR_SUM_FAST/QUANT_EXACT/QUANT_PACK/GEMV_SUM_FAST=1、SWIGLU_ABSMAX=0、GGML_KUNLUN_Q4_GATE_UP_PAIR未設定Auto。

## 再現

final-state.jsonのsource.finalのHEADと未commit差分を使い、新しい一意のworktreeにfinal-library.patch / final-llama.patchを適用する。既存完成版を上書きしない。production-only.patchは前回完成remaining-four-finalへの最小差分で、旧worktreeへは適用しない。manifestのsource SHAを照合する。

final/build-commands.jsonは実際の絶対パス。worktree/build/private installを新しい追試用ディレクトリへ置換し、Release/既存XRE/XDNN/XTDKを維持する。final/prebuiltを読取り使用するか、同じSDK/optionsでsourceから再生成してSHAを照合する。新旧P8/Q2 imageが使用imageとbyte一致する証拠はfinal/source-rebuild.json、全旧imageの保護はfinal/old-normal-images-preserved.json。

全環境はfinal/profiles.json、全引数はlogs/final-*.jsonを使う。通常の独立PP/TGは次の内容。

```bash
llama-bench -m EXISTING_MODEL.gguf -p 512 -n 128 -r 1 -b 512 -ub 128 -t 6 -ngl 99 -sm none -dev KUNLUN0 -mg 0 -ctk f16 -ctv f16 -fa on -o json
```

連続実行は-p 0 -n 0 -pg 512,128、LLAMA_BENCH_SPLIT_TIMING=1。KVを保持したTGと独立TGを別集計し、合成速度は640/実測PP+TG wallから求める。

このHostのbenchmarkだけLLAMA_BENCH_MAIN_CPU=2。main threadのみ固定しworkerは変更しない。別Hostでは許可CPU mask、SMT topology、XPU IRQ effective affinityを確認して指定する。IRQ/governor/clockを変更せず、mask復元ログを残す。

既存thermal guard、persistent latch、他利用者、両PD state/errno/errtask raw SHAとreset_countの監視を外さない。新しいresultsへrunnerをconfigureし、recovery-epoch.jsonとapproved-history rawを照合する。旧errtaskを新規異常へ無条件適用しない。原因不明の異常はfail-stop。今回resetは行っていない。

通常測定はLD_PRELOAD audit、inverse export、詳細traceを含めない。quant-audit.soは診断専用で、実際のMain/ResidualのK/dtypeとimageを集計する。idleのinverse imageも診断専用。stage単体の完了待ち付き計測は全体測定の後の別区間で、量子化全体/一連処理へstage間waitを追加しない。個別stageの時間を加算して通常全体の内訳としない。

各microは3独立process×31反復。final A/Dは各モデル・各workload3独立process。完了済みログ/markerは上書きせず、再実験は新resultsで行う。1process内の反復を独立nへ数えない。全採否はdecisions.jsonと総合REPORT.ja.mdを参照する。
