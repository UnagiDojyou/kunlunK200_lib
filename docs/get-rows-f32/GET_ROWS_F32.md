# F32 GET_ROWS on K200

GGML_KUNLUN_GET_ROWS_F32=1 enables F32 activation row selection on XPU. Unset, 0 or invalid values keep CPU F32 GET_ROWS. This setting is independent of GGML_KUNLUN_GET_ROWS (Q4 embedding) and COMMON_Q4. No existing defaults change.

GGML contracts: contiguous F32 matrix [K,N], no input/output views, K multiple of 16 in [16,1048576], N in [1,INT_MAX]; contiguous I32 vector [M] with 1<=M<=512; contiguous F32 output [K,M]. Unsupported dtype/stride/shape or split buffer uses existing CPU placement. Normal owning-PD activations produced during Layer/Row Split are supported.

Library API: get_rows_f32(Kernel&, device, output, output_bytes, input, input_bytes, indices, index_bytes, K, N, M). All pointers are device pointers on the chosen PD. Input/output need 64-byte alignment, indices 4-byte alignment, and the byte ranges must not overlap or overflow. Each Kernel is bound to one PD and calls serialize with the existing mutex. Caller keeps inputs/indices stable until return.

The CL kernel copies unsigned 32-bit words; NaN payloads, infinity and signed zero are preserved. Every worker validates all indices before any source-row access or output write. Invalid IDs return false without changing the output. The GGML backend fails safely for an invalid active ID, as CPU GET_ROWS does; it does not retry on the device.

Program and 256-byte status scratch persist with Kernel. No whole activation readback, new weight layout, output staging buffer, L3 allocation, requantization or MAC is used. Return occurs after kernel completion and status validation. Automatic tile is 64 elements for M=1, 1024 for other M; KUNLUN_GET_ROWS_F32_TILE=64|1024 is a diagnostic override, invalid values use auto.

Inactive GET_ROWS honors GGML_TENSOR_FLAG_COMPUTE. The existing inactive KUNLUN GET_ROWS copy skip retains all dependency waits. This change does not modify scheduler core, graph selection API, Q4/Q6 kernels, weight ABI or Activation Lease.

Corresponding new library and backend commits must be used together. The version string remains 0.7.12; older binaries do not export this API. For the measured baseline keep Q4 embedding ON, Q6_SDNN=1/LEGACY=0, SwiGLU=1 and Large PP=0.

This remains opt-in: large activation transfers and graph splits decrease, but overall PP/TG speedups are not established. See REPORT.ja.md for measured costs, numerical checks and limitations.
