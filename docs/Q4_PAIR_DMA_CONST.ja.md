# Gate/Up Pair DMA/controller定数化（既定OFF）

`KUNLUN_Q4_PAIR_DMA_CONST=1`で有効。未設定・0・その他の値はOFF。
Gate/Up Pair自体は従来どおり未設定Auto、`GGML_KUNLUN_Q4_GATE_UP_PAIR=0`で無効。
Preaddrは別機能で、`KUNLUN_Q4_PAIR_PREADDR=0/1`と独立に組み合わせられる。

| Preaddr | DMA_CONST | 経路 |
| --- | --- | --- |
| 0 | 0 | 正常Auto Pair |
| 1 | 0 | 既存Preaddr Pair |
| 0 | 1 | shape定数化 Pair |
| 1 | 1 | Preaddr + shape定数化 Pair |

対応は既存PairのF32/M1、4096x11008と2048x5632。元と同じgraph/alias/Lease/PD/L3 guardを維持。
非対応条件は従来のPair拒否・通常GEMV fallbackを使う。公開API・ABI・weight/scale layoutは変更しない。

既知shapeのcount16・stride等をコンパイル時に固定し、controllerのmin/乗算/loop準備を減らす。
DSの16回loopもコンパイラにより展開される。DMA本数・2D設定数・転送量は変わらない。
FP32順序、全fence/barrier、L1/L2/L3 bank所有権は維持する。
設定単独fence・DMA descriptor状態保持の省略は実装しない。

4つの新image（bin/entryの計8 files）が必要：
`kunlun-common-gemv-pair-dma-const-{nmajor,stream}`、
`kunlun-common-gemv-pair-dma-const-preaddr-{nmajor,stream}`。
一方だけのinstallや、通常版imageの名前で置き換える運用はしない。

ビルドは本resultsのbuild.jsonに記録した既存SDK Release構成を、新しいworktree/build/private install先で再現する。
normal-prebuiltには元の206 filesと新しい8 filesを揃える。通常installは上書きしない。
source状態はHEADだけで識別せず、final-state.jsonとproduction-only patch、image SHAを照合する。

通常benchmarkはcandidate-environment.jsonを使い、Preaddrの値を固定してDMA_CONSTだけを切り替える。
main affinityは既存helperに`LLAMA_BENCH_MAIN_CPU_POLICY=2`を渡す（内部でLLAMA_BENCH_MAIN_CPU=2を設定）。
CPU2はこのHost専用。別Hostでは許可CPU・SMT・IRQを確認して明示指定する。workerやIRQは変更しない。
COMMON_Q4=1、SUM_FAST=0、SWIGLU_ABSMAX=0、PD0、t6、b512/ub128、F16 KV、FA ONを使用。
通常独立PP512/TG128と、KVを保持する連続PP512→TG128は分けて扱う。

効果と採用上の制限はresults/gate-up-scale-resume-20261103/REPORT.ja.md参照。
少数processの差を一般的な改善率と解釈せず、既定値は変更しない。
