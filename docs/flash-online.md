# FlashAttention Online Softmax (0.7.1)

`KUNLUN_FLASH_ONLINE=auto` (also the unset default) selects the KL1 ordered-sum kernel. `legacy` selects the unmodified 0.7.0 kernel; unknown values also select legacy. `ordered` is an explicit alias for auto. This setting affects CL Online Softmax only.

The new kernel uses four independent scalar max chains, retains the original sequential FP32 sum and native `exp_s`, avoids finite unchanged-max rescaling, distributes short rows across all four clusters, and reuses a broadcast mask across heads for complete 64-query groups. F16 masks decode backwards in a shared packed/F32 LM buffer. Nonzero ALiBi forces mask reload to avoid multiplying cached values repeatedly. Each invocation invalidates its mask cache. SDNN launches and the score/probability/accumulator/stats L3 contract are unchanged.

The default block/query tile and TG GQA settings are unchanged. No additional profiling or waits are enabled by auto. The legacy CL kernel and the separate SDNN/CL attention routes remain available.

## Benchmarks

Build with `KUNLUN_BUILD_BENCHMARKS=ON` and the normal XRE/XDNN/XTDK variables. Examples (kernel directory is the build output, not the source directory):

```sh
# PD D block M H mask phase kernel-directory
build/k200-flash-online-bench 0 128 512 128 32 1 1 build/lib/kunlunk200/kernels
# PD D S M HK H pattern batch: auto vs 0.7.0 legacy plus CPU FP64
KUNLUN_FLASH_ONLINE=auto KUNLUN_FLASH_BENCH_SAMPLE=1 build/k200-flash-online-api-bench 0 128 2048 128 8 32 12
```

The standalone benchmark restores identical inputs outside each timed launch and reports launch+wait wall time, enqueue time and full-buffer bit comparisons. Mask 0 means none, 1 F16 causal, 2 F32 causal, 3 unmasked F32, 4 all masked, 6 NaN/Inf/extreme scores, 7 all 65,536 half bit patterns (M128/block512). Phase 0 initializes, 1 merges a previous block, 2 keeps an existing maximum of 100, 3 finalizes. L3 workspace must fit 16 MiB. The API benchmark checks all output elements against legacy; `SAMPLE` restricts only CPU FP64 reference queries. `KUNLUN_FLASH_FULL_Q_REF=1` includes the existing FP32-to-half Q rounding in the CPU error. `KUNLUN_FLASH_BENCH_MUTATE=1` checks same-pointer Q/K/V updates and input guards.

The rejected parallel-sum prototypes are under `bench/experiments/flash-online`; they are not built, installed, or dispatched by the library. They changed TinyLlama logits despite tiny attention-unit errors. See the Japanese report and its reproducibility scripts for the diagnostic kernels and results. The common device header uses `ONLINE_REDUCE=2`/`ONLINE_REUSE=1` in the shipping kernel; experimental macros do not change the shipping configuration.

Report: [Japanese evaluation](reports/flash-online-20261026/REPORT.ja.md).
