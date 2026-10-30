# Kunlun K200 Q6_K SDNN GEMV：オンチップ集約

2026-10-30 JST。**7Bの定常GEMVを4.557→2.837 msへ短縮し、CPUの3.481 msを上回った。FP32 partialのHBM workspaceとCL joinは除去した。** Tinyもlibrary呼出しは1.656 msへ改善したが、ホストへのlogits取得を含めるとCPUとほぼ同等だった。通常TG128は今回の両モデルでCPU経路を上回った。既定は引き続きCPU、SDNNはopt-inとする。

「llama.cpp開発」の現行Q6 SDNN版を継承し、独立branch `codex/k200-q6-onchip-20261030`、独立lib/llama worktreeで実装した。0.7.12安定版、前回SDNN版、Q4実装、SwiGLU ON・Large PP OFFを保護した。push・PR・main更新は行っていない。

既存版ではMAC結果をL1Eに置き、EWがK16 scale補正をしてL2Eへ出し、RS→L2R→DMAOUTで全partialをHBMへ保存していた。別CL kernelはtileごとのpartialをGM2LMで読み、FP32加算後にLM2GMで最終出力を書いていた。小さいN32/K256 panelごとのcontroller同期、多数のDS/MAC命令、およびこの往復と3回の起動が削減対象だった。エンジン別の詳細profileは追加していない。

採用方式は**A：EWで補正後のFP32 accumulatorをL1Eに保持する方式**。L1E容量に収まるため、L3を介するCL連携や段階的HBM集約は不要だった。

- K16のsigned W8A8 MACごとに対応するActivation chunk256 scaleとQ6 K16 scaleを適用し、その後にFP32累積する。異なるK16整数積和を単一scaleで補正していない。
- N32 tileを各clusterが独占し、K方向をN-majorで走査する。128組のK16 partialを16 KiBのFP32 accumulatorへ累積し、K全体の終了時だけ木状に縮約する。初回chunkは有効なpartialをコピーし、前回tileの値を読まない。
- K=2048/4096はK16×128のpanel。その他のlibrary対応KはK16×16へ切り替える。最終N32出力128 byteだけをRS→DMAOUTでHBMへ書く。
- W8のN16半分をL1Wの偶数/奇数byteへ配置し、SDK標準のN32出力配置を使う。MACの両出力bankを有効な列として使用し、K16あたり2回のMACを1回へ減らした。出力bank offsetをpanel長に合わせ、全32列の書込み領域を予約した。
- K16×16あたり32回だったweight DSを、新panel全体で2回へまとめた。DMA panelを8 KiBから64 KiBへ拡大し、同期反復数を7Bで4000→500回/clusterへ減らした。比較した改善はオンチップ集約、W8/DS配置、panel拡大の3種類。panel長は64と128を比較した。
- 8 controllerのDMA/DS/MAC/EW/RS担当、L1W/resultのdouble buffer、scaleの3 slot、resident Activationを維持した。依存するEW演算間のxfence、engine完了待ち、cluster sync、runtime_mutex、PD所有権検査を維持した。KL1で使えないevent APIは使っていない。

scaleのDSとEW消費には2段の間隔があるため、scale 3 slotは継続した。MAC、scale DS、EW scratchのL1E領域は重ならない。最大panelでのv16f32 indexはMAC `[0,512)`、scale `[512,1280)`、temp `[1536,1792)`、accumulator `[1792,2048)`。EWからL2Eへ出すのはtile最終chunkだけで、RS/DMAOUTもそのchunkだけを処理する。cluster間で同じN tileを更新しないため、跨cluster reductionはない。

| Q6用メモリ・論理転送（decimal MB） | 7B 旧→新 | Tiny 旧→新 |
|---|---:|---:|
| FP32 partial workspace | 32.768→0 | 16.384→0 |
| HBM workspace全体 | 33.031168→0.263168 | 16.515584→0.131584 |
| 除去したpartial書込み＋join読込み/呼出し | 65.536 | 32.768 |
| 最終出力 | 0.128→0.128 | 0.128→0.128 |
| W8＋FP32 scale常駐weight | 163.840→163.840 | 81.920→81.920 |
| kernel起動/row | 3→2 | 3→2 |

workspace全体は約99.2%削減。残るscratchはA8 Activationとそのscaleのみ。上記転送量はアドレス・命令から求めた論理量であり、HBMカウンタの実測値ではない。重みの10 bpw展開は変更していない。CPU側にはfallback用の元Q6 bytesを保持する。

最大panel使用時、L1Wは32→128 KiB、L2_0は16→128 KiB、L2_1はActivation stagingを含む最大16→32 KiB/cluster。L1Eの使用アドレス上端は22→128 KiBで、SDK容量256 KiB以内。L1Dは7B 256 KiB/Tiny 128 KiBのまま。L2E/L2Rのdouble-buffer用アドレス範囲は各8→32 KiB以内、実際の最終出力は各128 byte。Q6のL3確保は新旧とも0。各SRAMは別領域であり、固定容量を越える割当てはない。

実重み・同一sin Activation、M=1、PD0、CPU6 thread。pack/uploadと初回kernel loadを除外し、A8量子化・起動・出力HBM書込み・完了待ちを含む。旧→新→新→旧の対称順序で各arm 3回中央値を2標本取得し、その中央値を示す。CPUは4 armの中央値。

| 単体GEMV | CPU Q6×Q8_K ms | 旧SDNN ms | 新SDNN ms | 新/CPU時間 |
|---|---:|---:|---:|---:|
| 7B K4096×N32000 | 3.481 | 4.557 | **2.837** | 0.815倍 |
| Tiny K2048×N32000 | 1.693 | 2.521 | **1.656** | 0.978倍 |

集約のみの試作は7B約4.15 msだった。標準W8配置だけでは約4.12 msに留まり、panel拡大を合わせて約2.84 msになった。不要な詳細profileや全shape sweepは行わなかった。

128 KBの最終logitsのD2Hと完了待ちも加えた別測定は次のとおり（各3回中央値）。CPUは同じ新SDNN試験内の参照時間。**Tinyのホスト出力取得込みではCPU超過を明確には達成していない。**

| ホスト出力取得込み | CPU ms | 旧SDNN＋D2H ms | 新SDNN＋D2H ms |
|---|---:|---:|---:|
| 7B | 3.458 | 4.657 | **2.881** |
| Tiny | 1.687 | 2.577 | **1.693** |

通常llama-benchは同一binary/library、Single PD0、PP512/TG128、batch/ubatch=512/128、threads6、F16 KV、SwiGLU=1、Large PP=0、既存Q4最適化ON。CPU→旧→新→新→旧→CPUの3方式対称順序で各2標本。平均±標本標準偏差、token/s。profile・audit・KV保存はこの性能測定に含めていない。

| モデル/phase | CPU経路 | 旧SDNN | 新SDNN |
|---|---:|---:|---:|
| 7B PP512 | 80.291±0.133 | 80.046±0.131 | 79.957±0.009 |
| 7B TG128 | 8.473±0.0001 | 8.400±0.0002 | **8.530±0.004** |
| Tiny PP512 | 323.391±3.934 | 317.195±1.601 | 321.284±7.179 |
| Tiny TG128 | 27.750±0.254 | 27.885±0.221 | **28.874±0.009** |

今回のTG変化はCPU比7B +0.68%、Tiny +4.05%、旧SDNN比+1.55%/+3.55%。少数標本であり、他のshape・contextへ一般化しない。PPはCPU比−0.42%/−0.65%を観測し、高速化は主張しない。特にTiny PPのばらつきは大きい。Q4コードは変更せず、インストールした非Q6 kernel 89個のbinary一致を確認した。

| 新SDNN単体の誤差 | 7B | Tiny |
|---|---:|---:|
| 同じA8のF64参照へのNMSE | 7.532e-15 | 6.874e-15 |
| 同参照への最大絶対誤差 | 1.756e-7 | 1.331e-7 |
| CPU Q8_K参照へのNMSE | 4.056e-13 | 2.279e-13 |
| CPUへの最大絶対誤差 | 1.281e-6 | 8.717e-7 |
| 旧SDNNとのNMSE | 9.265e-14 | 4.294e-14 |
| 旧SDNNとの最大絶対誤差 | 8.345e-7 | 5.960e-7 |

全出力finite、前後256-byte guard、ゼロ入力、F16/F32同値入力のbitwise一致、M2の行stride・ゼロ2行目、PD1が合格。異なるchunk256の振幅を大幅に変える入力、K768の小panel、N32/160/320の不均等cluster分配も確認した。同一workspaceを旧→新→旧または新→旧→新に切替えても、戻した結果はguardを含めbitwise一致した。HBM partialも新経路へ戻る際に解放される。静的なSRAM範囲・所有権確認とこれらの実機試験を行ったが、未試験の全入力・全競合条件を形式的に証明したものではない。

PP512→greedy8とF16 KV保存/復元をCPU・旧・新で比較。新とCPUのlogits NMSEは7B 2.577e-14/Tiny 1.582e-14、旧とのNMSEは6.536e-14/2.309e-14。最大絶対差はCPU比3.338e-6/3.815e-6、旧比9.537e-6/7.629e-6。greedyは両比較とも8/8一致し、復元後のnext logitsは各構成内でbitwise一致した。TinyのActivation共有ON/OFFもbitwise一致。PP128→TG4ではPD1とRow Splitも合格し、非対応descriptor・view・flag OFF/未設定のCPU fallback policyも通った。品質差が小さいため大規模PPLは追加していない。

HBM peakは別のPP512→TG8/KV復元smokeで既存allocation auditを用いて測定した。定常性能表とは混ぜていない。

| tracked HBM peak（decimal MB、PD0） | CPU | 旧SDNN | 新SDNN |
|---|---:|---:|---:|
| 7B | 4288.926 | 4485.810 | 4453.044 |
| Tiny | 656.353 | 761.221 | 744.839 |

旧比で32,765,440 / 16,381,440 byte減少。partial削減量との差2,560 byteはkernel code allocation差に対応する。重み常駐分があるためCPUより総HBM量は増える。PP→TGでweight uploadは1回のまま、ActivationのCPU readbackは0。各rowのjoin_launches=0を実LLMログでも確認した。

変更はlibraryの `kernels/kunlun-q6-onchip.xpu`、`src/q6-k.inc`、CMake登録、APIの診断counter追加、既存単体test拡張。GGMLの既存typed dispatchはそのまま新libraryを呼び、終了時counterを新旧の実起動数へ修正した。旧kernel・旧join・Q4 ABI・Activation leaseは維持した。

```sh
# 既定/0はCPU。対応する新libraryとkernelを使う。
KUNLUN_Q6_K_SDNN=1 KUNLUN_Q6_K_LEGACY=0   # 新オンチップ版
KUNLUN_Q6_K_SDNN=1 KUNLUN_Q6_K_LEGACY=1   # 旧HBM partial版
KUNLUN_Q6_K_SDNN=0                        # CPU
```

GGMLの対象条件は従来どおりK2048/4096、N32000の実検証済みQ6出力層。未設定/0、非対応shape/stride/view等は既存fallback。全shapeへの既定有効化は行っていない。

実機rootは `/home/unagidojyou/kunlun-k200`。worktreeは `worktrees/q6-onchip-20261030-{lib,llama}`、buildは `build/q6-onchip-20261030-{lib,llama}`、独立install・少量JSON・runner・ログは `results/q6-onchip-20261030`。`llm-bench.py`、`llm-smoke.py`、`final-check.py`、`regression.py`を既存thermal guard経由で再実行できる。新kernelはXTDKの既存compile-native.shでsourceからbuildした。Git HEAD・gitlink・binary hashは `delivery-summary.json` / `summary.json`を参照。

最大観測60℃。両PD RUNNING、errinfo/errtaskは開始時から不変。各実行前の他利用者確認、thermal guard、abort latchを維持し、reset・driver・firmware・clock・OS変更は行わなかった。残る候補は量子化と起動の固定費、128 KB logits D2H、K16ごとのMAC発行費、およびW8重みのHBM量。次はこれらを絞って改善する。必要なfenceの削除やQ4経路の変更は行わない。
