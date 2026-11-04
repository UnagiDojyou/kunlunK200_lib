# Gate/Up Pair address準備の前倒し（独立opt-in）

`KUNLUN_Q4_PAIR_PREADDR=1`で有効。未設定、0、不正値は従来Pairへ戻る。Gate/Up Pair自体の未設定Auto動作と`GGML_KUNLUN_Q4_GATE_UP_PAIR=0`による無効化を維持する。既存の`GGML_KUNLUN_COMMON_Q4=1`の前提も変えない。

対象はF32/M1・共通Q4_0 ABI-v1のGate/Up、K4096/N11008とK2048/N5632。既存graph検出、Lease、alias/PD/shape/dtype/configurationのguard、L3不足時のfallbackはそのまま。非対応graphでは通常の個別GEMVへ戻る。新opt-inをOFFにすれば従来Pair imageを使用する。

DMAIN0/1の次taskのweight/scale addressを、既存engineのxfence待ちに入る前に計算する。元と同じwide pointer計算とhigh/low抽出で、high bitsやcarryを切り捨てない。privateなvolatile整数4個は、計算をfence前に残すためのcontroller-local storageであり、共有bufferの書込みや新しいhardware fenceではない。

DMA descriptor設定とcommand発行は従来どおり、既存xfence完了・owner sync・engine lockの後に実行する。転送粒度、転送量、順序、double buffering、DS/MAC/EW/RS/DMAOUT、K32 scale補正、FP32演算順序、Gate/Up境界のdrain、resident Activation preloadを変更しない。

新しいdata HBM/L3領域やweight cacheを追加しない。controller stackのprivate address値が増えるため、生成assemblyでframe・code容量を確認する。通常kernelは112-byte ABIのまま。step trace用の640-word/core領域は診断imageだけで、通常imageに含まれない。

## Build・load

既存正常版を上書きせず、独立build/installを使う。CMakeの既存kernel targetへ次の2 imageを追加する。

- `kunlun-common-gemv-pair-preaddr-nmajor.bin/.entry`
- `kunlun-common-gemv-pair-preaddr-stream.bin/.entry`

`common-q4/gemv-pair-preaddr.h`を既存XTDK2.7.98.2・Release条件でbuildする。Host libraryは内部Program slot27/28を使い、公開APIとStatistics ABIは変えない。`KUNLUN_KERNEL_DIR`と`LD_LIBRARY_PATH`は同じprivate installを指す。prebuilt modeでも従来imageとこの2 imageを揃える。足りないimageはload errorであり、非対応shapeのfallbackとは別である。

実体は`results/gate-up-critical-20261103/final-state.json`で識別する。HEADだけでなく未commit patch・source SHA・使用image・DSO・benchmark binaryを照合する。

## 測定

既存80℃ thermal guard、persistent abort latch、他利用者確認、両PD監視を通す。通常性能ではprofile/trace/SDK timingを無効にする。

`LLAMA_BENCH_MAIN_CPU=2`は今回のi7-8700専用設定で、通常推論で自動固定しない。他Hostでは標準affinity helperでtopology/IRQを確認する。PairはAuto、COMMON_Q4=1、SUM_FAST=0、SWIGLU_ABSMAX=0、他はAuto基準profile。OFF/ONを同一binary/library・同じCPU配置で順序反転して比較する。

通常TGの採用判断、少標本による限界、step traceの計測負荷はREPORT.ja.mdを参照する。raw SDNN、量子化＋直接SDNN＋CL sumのprobe、実際のPair API、通常LLM wall時間を混同しない。
