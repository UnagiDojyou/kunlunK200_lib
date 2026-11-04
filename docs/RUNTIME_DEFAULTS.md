# K200 runtime defaults (2026-11-04)

The following validated paths are enabled when their environment variable is unset. Set a variable to 0 to disable that feature; 1 enables it explicitly. Other values disable it, except QUANT_PACK_EXT also accepts 5632 or 11008 for a single added K.

- KUNLUN_Q4_PAIR_PREADDR
- KUNLUN_Q4_PAIR_DMA_CONST
- KUNLUN_Q4_PAIR_SUM_FAST
- KUNLUN_Q4_QUANT_EXACT
- KUNLUN_Q4_QUANT_PACK
- KUNLUN_Q4_QUANT_PACK_EXT
- KUNLUN_Q4_GEMV_SUM_FAST

Existing shape/dtype/M guards and fallbacks still apply. QUANT_PACK requires QUANT_EXACT, and QUANT_PACK_EXT requires both. K4096 remains Q2. K2048 retains P8 independently of EXT. Pair uses graph Auto detection. SwiGLU absmax remains OFF. Kernel arithmetic, ABI, buffer lifetime and synchronization are unchanged.

Build library and images with the existing XRE/XDNN/XTDK, install to a private prefix, then build llama.cpp against that prefix. Use LLAMA_BENCH_MAIN_CPU only as an explicit benchmark opt-in (CPU2 on the validated i7-8700 host); workers and system IRQ/governor are not changed.

Earlier experiment documents describe historical opt-in defaults. This document supersedes their default-value statements. Performance evidence and raw reproduction artifacts remain under results/activation-two-20261104 and its archive.
