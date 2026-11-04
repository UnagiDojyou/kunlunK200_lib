# Ordered Q4 GEMV CL sum opt-in

`KUNLUN_Q4_GEMV_SUM_FAST=1` enables a mode-1-only CL partial reduction for exact
`(K,N)` pairs `(4096,4096)`, `(4096,11008)`, `(11008,4096)`, `(2048,256)`,
`(2048,2048)`, `(2048,5632)`, `(5632,2048)` when the existing stream route is active.
Unset, `0`, invalid values, padded inputs, unsupported shapes, or
`KUNLUN_Q4_MAC_DMA=0` keep the previous sum. M>1 is unchanged.

The original mode-0/mode-2 sum and all existing images remain available. The new
kernel specializes already-scaled K32 partials, uses bounded unsigned offsets,
and reads 16 rather than 8 K blocks per DMA. Every output retains the same
increasing-K FP32 add sequence. No activation, weight ABI, SDNN kernel, L3
layout, buffer lifetime, launch count, or synchronization changes are made.
The new CL kernel uses the existing 4-cluster / 16-core launch configuration.
GEMV address32 and Tiny GEMM Owners keep their independent flags.

The flag remains OFF by default. Local TG128 tests improved, but normal PP
measurements showed small differences and a Tiny PP-only run showed a larger
slowdown. Kernel-code allocation and process-level timing modes remain unresolved.
Use the report's paired results; do not assume its median speedup is universal.
