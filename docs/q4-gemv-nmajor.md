# Optional N-major GEMV (0.7.9)

Set KUNLUN_Q4_MAC_DMA=1 and KUNLUN_Q4_NMAJOR=1 with GGML_KUNLUN_COMMON_Q4=1. The new switch defaults to 0. It selects the N-major kernel only for M=1 and padded (K,N) pairs (4096,4096), (4096,11008), (11008,4096), after the existing ABI/shape checks. Unsupported shapes use the existing path; invalid settings log once and use the 0.7.8 stream order.

The traversal handles both adjacent N64 halves and then advances the resident K window before the next N128 group. The existing t-indexed buffers, scale caches, pipeline delays, fences, and barriers are unchanged. Each K32 result is written to its original partial address; the final CL addition order is unchanged.

The public API, weight ABI, Activation quantization/Lease, GEMM, and Attention are unchanged. No weight repack or new data workspace is needed. The fixed-mode code allocation remains 8448 bytes. Switching both modes in one workspace retains both programs and adds 8448 bytes.

Focused PD0 tests found a small 7B TG gain (about 0.8% in the initial trial, 2.0% in the final trial). TinyLlama did not benefit with unrestricted N-major, so its shapes are excluded. Row Split representative shapes also retain the previous path. Default behavior is unchanged. See the focused Japanese evaluation bundled with this task for samples, precision checks, and limitations.
