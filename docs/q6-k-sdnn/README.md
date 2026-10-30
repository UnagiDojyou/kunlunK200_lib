# Q6_K SDNN GEMV（0.7.12を基準とした実験経路）

Q6_K出力層をCPUへ戻さず、load時のlossless signed-W8 packとSDNN W8A8 MACで計算する。新しいEW/L1E集約は検証済み7B GEMVと両モデルのTGを改善したが、既定は引き続きCPU。Tinyのホストlogits取得込みはCPUとほぼ同等。

- backend: KUNLUN_Q6_K_SDNN=1 で有効。未設定、0、無効値では従来CPU経路。
- GGML: contiguous matrix、Q6_K weight K=2048/4096、N=32000、contiguous F32 activation/output、M=1..512。viewは非対応。
- M>1はGEMVの逐次反復であり、高性能なQ6 GEMMではない。M>512やその他のweight shapeはCPU fallback。
- Row Splitの境界だけ32行単位へ切下げる。各PDの行数は32の倍数で、出力順序は元の順序を保つ。
- Library API: Q6KWeight、Q6KWorkspace、q6_k_matmul。K=256..4096（256の倍数）、N=32..32000（32の倍数）。F32/F16、contiguous input、64-byte aligned output、非alias、有限入力を必要とする。
- WeightType::q6_k は識別子。従来matmulのraw-pointer契約へ渡さず、型付きQ6KWeight APIを使用する。
- weightとworkspaceは別所有。1 tensor/owning PDにつきpack allocation 1個、uploadはload時のみ。生Q6をHBMへ二重常駐させない。CPU fallback/get_tensor用に元のGGML bytesをhost側へ保持する。
- Q4のABI v1と本Q6 pack v1は別物。Q6KWeightのready・ABI・PD・shape・pointer範囲を検査してから起動する。
- Q6のK16 signed scaleとFP16 dをFP32へ正確に保持。W8 packは8bpw、FP32 scaleは2bpw、計10bpw（元Q6_Kは6.5625bpw）。
- ActivationはXPUでchunk256ごとに対称A8へ量子化する。Q4のLease、量子化方式、重み配置は変更しない。Q4 Leaseとのbuffer共有は行わず、Q6専用workspaceを再利用する。
- 新経路はQ6専用量子化CLとSDNN（8 controller、4 cluster）の2起動。K16補正後にL1EでFP32集約し、最終出力のみHBMへ書く。default-stream順序と完了待ちを維持する。
- KUNLUN_Q6_K_LEGACY=1 は旧HBM partial＋joinの3起動を診断用に選択する。未設定/0は新経路。
- SDNNはL1D Activation常駐（7B 256KiB、Tiny 128KiB）、K16/N32のW8 DMA、独立K16 scale補正、N-major走査。SDK標準W8の32列を両出力bankで利用し、MAC/scale/temp/accumulatorの領域を分離する。
- 最終logitsのCPU sampling用D2Hは残る。最終hidden stateの読戻しが消えても、全logits取得ではD2H総量は増える。

必要な新APIを含むlibrary commitと、このbackendを組み合わせること。同じ0.7.12というversion表示でも、変更前のDSOにはQ6 APIがない。通常のsubmodule buildか、対応するsystem-library installを使用する。

詳細な数値・採否・再現方法はREPORT.ja.mdを参照。SDK、モデル、kernel binary、測定ログはcommitしない。

現在の性能・workspace・精度・旧版切替は[オンチップ集約の結果](../q6-onchip/REPORT.ja.md)を参照。以前のREPORTは旧SDNN版の記録として残す。
