# Q4 GEMV MAC/DMA stream (0.7.8, opt-in)

KUNLUN_Q4_MAC_DMA=1 enables the validated stream GEMV for common Q4 weights. The default is disabled. 0 selects the stable kernel; other values log a reason and use the stable kernel. Legacy native Q4, CDNN and direct Q4 do not use this path. Attention remains at the 0.7.6 kernel implementation.

The public common weight ABI v1, original FP16 K32 weight scales and two-component A8 quantization are unchanged. Both common_q4_matmul and common_q4_matmul_shared dispatch through the same GEMV selector. Activation Lease identity, generation, buffer ownership and lifetime rules are unchanged. M>1 retains the existing Combined/Wide/Rows/Combo kernels.

## Selection and fallback

Only M=1 with K=Kp and these tested padded shapes uses the stream kernel:

| K | Np |
| --- | --- |
| 2048 | 256, 1024, 2048, 2816, 5632 |
| 4096 | 256, 2048, 4096, 5632, 11008 |
| 5632 | 1024, 2048 |
| 11008 | 2048, 4096 |

Logical N may be smaller than Np; existing output cropping is retained. Other shapes use the stable GEMV and increment shape_fallbacks. The existing L3 capacity check and exact CL fallback are unchanged. Invalid weight ABI, dtype, alignment and overlap are checked by the existing APIs before dispatch. Kernel errors retain the fail-fast behavior and must not be hidden by retries.

## Schedule and memory

The same four clusters and eight engine-owner cores are used. A resident group contains at most 128 K32 Activation blocks. It occupies at most 512 KiB of L1D, rather than the previous 64 KiB window. Each resident group is completely loaded before use and all engines drain before it is replaced. K512 windows within the group share one continuous pipeline.

DMA, DS, MAC, EW, RS and output instructions retain their engine ownership and existing completion fences. Weight double buffers, two scale caches and three L1E scale slots are unchanged. Template owner functions move core selection out of the inner loop and reduce unrelated register liveness. Each core still executes the same number of cluster barriers. The SDK sync primitive synchronizes all cores in a cluster.

MAC order for each K32 partial remains main-component MAC followed by residual-component accumulation, then the original weight-scale EW and RS. Partial storage and the final CL summation order are unchanged. The experimental scale-configuration hoist is not enabled in production.

No new weight pack, quantization, HBM staging, transfer or host wait is added. L1W/L1E/L2 and L3 usage are unchanged. The new program is 7960 bytes; the existing loader allocates 8448 HBM bytes per workspace/PD when first used. L1D is kernel-local SRAM use, not a persistent HBM allocation.

CommonQ4Workspace::mac_statistics() returns separate baseline_sd_calls, stream_sd_calls, shape_fallbacks and peak resident_l1d_bytes. The layout of the existing CommonQ4Statistics and the pointer-sized public workspace/Lease classes is unchanged. These are path counters, not device busy-cycle measurements.

## Build and verification

A normal source build produces 86 kernel images. All 85 pre-existing .bin/.entry pairs were reproduced byte-for-byte from 0.7.7. The new image matches the tested special128 candidate. The prebuilt-kernel option requires the complete new image set; mixing old directories that omit the new image is rejected by CMake.

k200-q4-mac-bench PD K N M f32|f16 pattern trials compares stable/stream, ordinary/prepared API execution with reversed orders, warmups and raw samples. It verifies output guards, bitwise output equality, unchanged quantization/scales and weight payload, and one weight upload. It also checks invalid-mode fallback. pattern 0 is normal, 1 is outliers, 2 is zero.

Example, under the existing thermal guard:

~~~sh
k200-q4-mac-bench 0 4096 4096 1 f32 0 15
k200-q4-mac-bench 1 11008 4096 128 f16 0 7
~~~

Diagnostic profiles and assembly are kept in the separate mac-dma-20261029 result directory. Device-clock issue, completion and barrier intervals overlap across owners; do not add them to obtain kernel wall time. Normal performance runs do not use diagnostic kernels or additional GPU synchronization. Adoption is based on end-to-end LLM measurements documented in the final report; this feature remains opt-in.
