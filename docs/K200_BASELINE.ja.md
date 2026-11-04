# K200 Auto開発基準（2026-11-03）

`results/gate-up-auto-20261103/final-state.json`をsource・未commit累積patch・kernel image・DSO・benchmark binaryの識別に使う。HEADだけでは正常版を再現できない。両worktreeのbranchは`codex/k200-gate-up-auto-20261103`。前回の正式opt-in基準と実験用worktree/install/resultsを保持する。

## Gate/Up Pairの設定

| GGML_KUNLUN_Q4_GATE_UP_PAIR | 動作 |
|---|---|
| 未設定 / `auto` | Auto。既存graph判定を満たす場合だけPairを実行 |
| `1` | Autoと同じ対応経路を有効化（従来互換） |
| `0` | 無効、従来の2 GEMV |
| その他 / 空文字 | 安全側として無効、従来GEMV |

既存の前提`GGML_KUNLUN_COMMON_Q4=1`は引き続き必要。Pairの既定Auto化は他機能の既定値を変更しない。「新基準」はsourceと明示されたbenchmark profileの組であり、全環境変数を未設定にして全最適化が有効になる意味ではない。

F32・M1、共通Q4_0 ABI-v1、K4096/N11008またはK2048/N5632、共有Activationの隣接MUL_MATとSwiGLUが対象。別shape/dtype/PD、split/view/alias、graph分割、出力の複数consumer、不適合Lease/config、L3不足は既存経路へfallbackする。入力の追加consumerは既存どおり許容。PP512にも最後のM1 FFNがありPairの対象となる。詳細はGATE_UP_PAIR.mdを参照。

kernel・重みABI・Activation共有Lease・2出力管理・FP32累積順序・fence/barrier・buffer寿命は前回版と同一。通常画像とPair画像は既存の検証済みimageをSHA照合して再利用する。

## 既定値と実際の測定設定

| 機能 / 環境変数 | sourceの未設定時 | 今回のbenchmark profile |
|---|---|---|
| GGML_KUNLUN_Q4_GATE_UP_PAIR | Auto（今回変更） | 未設定 / `0`を対照 |
| GGML_KUNLUN_COMMON_Q4 | OFF（従来どおり） | `1` |
| KUNLUN_Q4_DMA_ADDR32 | OFF（従来どおり） | `1` |
| KUNLUN_Q4_GEMM_OWNERS | 対応7B M128はAuto | `1` |
| KUNLUN_Q4_GEMM_TINY_OWNERS | OFF（従来どおり） | `1` |
| GGML_KUNLUN_Q4_ACT_SHARE | 対応条件で有効 | `1` |
| KUNLUN_Q4_GEMV_SUM_FAST | OFF | `0` |
| GGML_KUNLUN_SWIGLU_ABSMAX | OFF | `0` |
| GGML_KUNLUN_Q4_GATE_UP_PAIR_STATS | OFF | 通常性能`0`、dispatch診断`1` |
| LLAMA_BENCH_MAIN_CPU | 無固定 | `2`（本Host専用指定） |

Attention/FlashAttention、RoPE、SwiGLU、Q6_K SDNN、Embedding/F32 GET_ROWS、mixed pruningなどは既存正常版を継承する。全ての実際の測定値は`candidate-environment.json`とrun別のenvironment/argvへ保存する。Main+Residual統合試作は含めず、既存Max/Main/Residualの経路を維持する。

## Benchmarkと再現

既存のthermal guard・persistent latch・他利用者確認・両PD監視を備えた`tools/llama-bench/kunlun_guarded_runner.py`を使用する。`LLAMA_BENCH_MAIN_CPU_POLICY=2`をrunnerへ渡すと、benchmarkの既存worker作成・warmup後にmain threadだけを固定し、計測後に元maskへ復元する。CPU2は他Hostへハードコードしない。auto候補の確認は`python3 tools/llama-bench/bench_affinity.py --main-cpu auto --inspect`。IRQ/governorは変更しない。

今回の性能smokeはPD0、PP512/TG128、threads6、batch512/ubatch128、F16 KV、FlashAttention ON、同一binary/libraryのPair未設定と`0`。各条件1独立process、process内2反復に限定する。改善率の再推定には使わず、前回6独立processの正式評価を主な性能根拠とする。

記載HEADへ`library.patch` / `llama.patch`を適用するか、前回正式opt-in treeへ`*-auto-only.patch`を適用する（二重適用しない）。`build.json`の条件で新しいbuild/installを作り、`candidate-environment.json`の新libraryとkernel directoryを使う。既存installを上書きしない。immutableなsource一式、patch、使用環境、生成DSO/benchmark/imageのSHAはfinal-state.jsonとartifact manifestで照合する。
