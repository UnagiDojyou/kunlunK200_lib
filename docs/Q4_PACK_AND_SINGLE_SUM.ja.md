# Q4_0 remaining-four 再現手順と追加opt-in

成果物はresults/remaining-four-20261104。基準はfive-stage-20261104の完成済みprofile C。

## 実行profile

final/profiles.jsonのA/P/S/Cを使用する。共通でCOMMON_Q4=1、Pair Auto、PREADDR=1、DMA_CONST=1、PAIR_SUM_FAST=1、QUANT_EXACT=1、SWIGLU_ABSMAX=0。Aは前回Cと同じ最適化状態を今回の同じbinary/DSOで再測定した条件。

| 条件 | KUNLUN_Q4_QUANT_PACK | KUNLUN_Q4_GEMV_SUM_FAST |
| --- | --- | --- |
| A: 前回C | 0 | 0 |
| P: pack単独 | 1 | 0 |
| S: 既存single sum単独 | 0 | 1 |
| C: 統合 | 1 | 1 |

QUANT_PACKは既定OFF、未設定/0/不正値で旧Q2。QUANT_EXACT=1、F32 M1、K=2048のときだけ追加packを使用する。K=4096/5632/11008、F16、M>1は従来経路。PREADDR/DMA_CONST/PAIR_SUM_FAST/QUANT_EXACT/SUM_FASTの既定値は変更しない。Pair本体のGGML_KUNLUN_Q4_GATE_UP_PAIRは未設定/auto/1で既存条件によるAuto、0で無効。

新imageはkunlun-common-main-packとkunlun-common-residual-packのbin/entryだけ。Maxは既存exactを共有する。SUM_FASTは既存16-block imageを使い、新しい32-block版は実験用としてのみ保存する。FP32演算順序、量子化式、ABI、同期、Leaseは変更しない。

## source・buildの復元

final-state.jsonのsource.finalのHEADと未commit差分を使い、一意な名前の新worktreeにfinal-library.patch / final-llama.patchを適用する。HEADだけで状態を識別しない。source manifestとpatch SHAを照合する。production-only.patchは前回完成Cに対する最小差分で、保護対象の旧worktreeへは適用しない。

final/build-commands.jsonは実際の絶対パスの記録。worktree/build/private installのパスを新しい追試用ディレクトリへ置換して実行する。既存の正常版・既存build/installを再利用して上書きしない。Release、既存XRE/XDNN/XTDK、SDKコンパイル条件を維持する。final/prebuiltは読取り参照可能。sourceから再生成したimageと使用imageの一致はfinal/source-rebuild.jsonに記録。

## 安全なbenchmark

既存80℃ thermal guard、persistent latch、他利用者確認、両PD状態、reset_countと旧errtask raw SHAの監視を外さない。新しい異常を旧履歴として承認しない。追試用resultsへrunnerをconfigureし、check_users.py/recovery-epoch.json/旧履歴rawを照合してから起動する。失敗した候補を無条件再実行しない。

このHostだけLLAMA_BENCH_MAIN_CPU=2。main threadのみ固定し、workerは変更しない。別Hostは許可CPU集合、/sys/devices/system/cpuのSMT topology、/proc/interruptsとIRQ effective affinityを確認して指定する。IRQ/governorは変更しない。before/applied/measured_end/restoredログを保存する。

通常の独立PP/TG引数:

```bash
llama-bench -m EXISTING_MODEL.gguf -p 512 -n 128 -r 1 -b 512 -ub 128 -t 6 -ngl 99 -sm none -dev KUNLUN0 -mg 0 -ctk f16 -ctv f16 -fa on -o json
```

連続実行は-p 0 -n 0 -pg 512,128とLLAMA_BENCH_SPLIT_TIMING=1。KVを保持する連続TGと独立TGを区別する。全環境はfinal/profiles.json、実引数・CPU/温度/状態はlogs/final-*.json等を参照。通常性能へLD_PRELOAD/file-open auditや追加traceを入れない。

各条件3独立process。P単独は対象Kを持つTinyだけ測定し、7Bでは非対象なので省略した。統合Cの7B測定は実施し、P非発動をdispatchで確認する。同一process内の31micro反復を独立nとして扱わず、遅いprocessも除外しない。完了markerと既存ログは上書きせず、再実験は新しいresultsで行う。

## 診断・未採用候補

theme1はP4/P8とinverse直接bit比較、theme2は旧8/既存16/試作32-block、theme3はC/E image同一性と通常wall、theme4はD0-onlyと64-byte配置を保存する。いずれも正常版へ混ぜない。過去のassembly transplant、2D config設定保持/省略、fence削除、1-core量子化統合は再試行していない。

inverse診断はres_peaksの未使用領域とparameters[4]へ値を出す独立imageで、通常imageを変えない。診断clockやauditによる時間を通常性能へ混入させない。採否はdecisions.jsonと総合REPORT.ja.mdを参照する。
