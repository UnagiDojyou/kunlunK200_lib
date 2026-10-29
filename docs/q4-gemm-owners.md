# Optional M128 GEMM owners (0.7.10)

Set GGML_KUNLUN_COMMON_Q4=1 and KUNLUN_Q4_GEMM_OWNERS=1. The new option defaults to 0. Keep KUNLUN_Q4_MAC_DMA=1 and KUNLUN_Q4_NMAJOR=1 for the existing GEMV optimizations. Activation sharing is independent and unchanged.

The new kernel is selected only for physical M128 and padded K/N pairs (4096,4096), (4096,11008), (11008,4096), after existing shape/capacity checks. Other M or shapes keep the existing kernel. Invalid settings log once and fall back. Existing padding, crop, and M>128 chunk handling remain in the library.

Six dedicated controller functions handle DMA0, DMA1, MAC, DS0, DS1, EW. The other two controllers execute the same pipeline barriers before handling RS and output. M128/nt32/kt128 and work_m=1 are constants. The original compact-scale L1E layout, FP32 correction/accumulation, K slices, fences, and barriers are unchanged. The L1D budget is 512 KiB; the Rows128 L1E plan is 214528 bytes. No extra data workspace, weight repack, duplicate weight, or Activation quantization is introduced.

Focused tests found a clear 7B PP improvement. TinyLlama's unrestricted candidate showed a possible TG regression in two series, so those shapes are excluded despite its PP gain. Row Split representative shapes also fall back. See the Japanese evaluation for sample sizes and limitations. No default change or GitHub publication was made.
