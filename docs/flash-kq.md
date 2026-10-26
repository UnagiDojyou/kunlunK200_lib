# FlashAttention KQ reuse (0.7.2)

The existing `kunlun-flash-kq` is retained unchanged. `kunlun-flash-kq-reuse1024` holds the assigned K block in L1W and each quantized Q tile in L1D across key tiles. Input DMA/INT16 scales and per-output MAC arithmetic are unchanged. Output packets are capped at 128 KiB to fit both L2E and L2R. PP rows128 therefore use at most 256 output keys per packet; TG can use up to 1024.

`KUNLUN_FLASH_KQ=auto` (also unset) selects reuse when there is more than one 32-key tile per SDNN part; otherwise it retains legacy. `legacy` forces 0.7.1 KQ, `reuse1024` forces the new kernel, unknown values use legacy. `KUNLUN_FLASH_ONLINE=auto` remains the 0.7.1 ordered-sum Online Softmax. The existing FlashAttention opt-in flags remain necessary. The separate Attention `auto/sdnn/cl` and GEMV fallback policies are unchanged.

K input conversion is split into requests of at most 128 KiB, independently of the larger L1W allocation. With fewer KV heads, each head is divided into contiguous 32-key parts to use the four SDNNs. Q/K are not cached across API calls or KV updates. Q does not survive the existing KQV L1D contract across KV blocks. GQA head normalization and restoration are unchanged.

Build targets `k200-flash-kq-bench` and `k200-flash-kq-api-bench` compare raw KQ or the complete Flash API against legacy. The former takes `PD D S_source M HK Hq kernel-directory [pattern batch query-tile [suffix]]`; it measures the final PP512/TG1024 key block, not the whole source context. The API target takes `PD D S M HK Hq pattern [batch]`. `KUNLUN_FLASH_BENCH_SAMPLE=1` restricts only CPU FP64 reference queries, not old/new bit comparison.

The 32/128/256-key prototypes under `bench/experiments/flash-kq` are comparison sources only, not installed/dispatched kernels. Normal inference has no extra clock instrumentation or wait. See [Japanese report](reports/flash-kq-20261026/REPORT.ja.md) and its [reproduction instructions](reports/flash-kq-20261026/REPRODUCE.ja.md) for measurements, accuracy, limitations, diagnostic kernels and complete scripts. Small PP end-to-end gains vary and should not be inferred from the much larger KQ-only speedup.
