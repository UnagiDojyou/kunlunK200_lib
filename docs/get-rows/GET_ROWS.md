# Q4_0 Embedding GET_ROWS

GGML_KUNLUN_GET_ROWS=1 enables XPU row gathering with the existing common Q4 ABI v1 pack. Unset, 0 or invalid values retain CPU GET_ROWS. GGML_KUNLUN_COMMON_Q4=1 is required. This remains opt-in: current PP/TG changes are small and TG regresses slightly.

The model loader prefers the device of the first offloaded layer for the Q4 token embedding. Partial offload with the first layer on CPU retains CPU input. Unsupported input tensors still use CPU. In Row Split the embedding uses one normal owning-PD buffer; a tied output without output.weight retains CPU input to avoid another full table beside the split output pack. The tested GGUFs have separate Q4 embedding and Q6 output weights.

Supported GGML contracts: Q4_0 contiguous matrix weight, no weight view, common layout, K divisible by 64, CommonQ4 supported dimensions, contiguous I32 vector indices M=1..512, contiguous F32 output [K,M]. Raw/legacy-native/I8 layouts and F16/F32 weight gathers are not added. Unsupported descriptors use CPU.

Library API: common_q4_get_rows(Kernel&, device, output, output_bytes, CommonQ4Weight&, indices, index_bytes, rows). All pointers are device pointers. The typed weight must be uploaded, ABI=1/layout=1/scale_order=1, on the selected PD. Output is 64-byte aligned; indices are 4-byte aligned. Input/output/weight ranges must not overlap. One Kernel is bound to one PD. Calls serialize on runtime_mutex and complete before return. Invalid descriptors return false before launch. Invalid IDs are detected before weight access/output writes; no output is written. The backend rejects invalid active IDs just as CPU GET_ROWS does.

The CL kernel reads two adjacent 16-bit banks (64 bytes) and the K32 FP16 scale, decodes the signed Q4 pairs, and writes original F32 dequantized values. It uses no SDNN MAC, weight requantization, persistent F16/F32 weight expansion, activation cache, or L3 allocation. The existing CommonQ4Weight allocation is reused, including when the same GGML tensor serves multiple operations. CPU fallback retains the original host bytes through existing transfer code.

Automatic chunk: M>=128 with K divisible by 256 uses 256 elements/job; otherwise 64. KUNLUN_GET_ROWS_TILE=64 or 256 is a diagnostic override; invalid/unavailable values retain automatic selection. A 256-byte status buffer and cached kernel image are reused. No per-token weight pack or upload occurs.

Non-selected graph branches must not execute GET_ROWS on uninitialized mixed-token IDs. The backend follows GGML_TENSOR_FLAG_COMPUTE for GET_ROWS. The scheduler omits data copies of inactive KUNLUN GET_ROWS, while retaining existing dependency waits. Diagnostic callbacks also ignore inactive nodes.

Q6 configuration remains separate: KUNLUN_Q6_K_SDNN=1 and KUNLUN_Q6_K_LEGACY=0 select the on-chip baseline. SwiGLU=1 and Large PP=0 remain unchanged. Corresponding updated library and backend commits must be used together; the version string remains 0.7.12, but older binaries lack the added API.

See REPORT.ja.md for measurements, limits and build commands.
