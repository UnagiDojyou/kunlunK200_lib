# Q4 Gate/Up pair (default Auto)

Unset `GGML_KUNLUN_Q4_GATE_UP_PAIR` or set it to `auto` to automatically pair eligible Gate/Up nodes in the normal K200 backend. `1` is equivalent for compatibility. Set `0` to disable pairing and use ordinary GEMV. All other values, including an empty string, safely disable pairing. Auto does not force unsupported graphs or shapes. Use the matching library and all three pair images; do not replace an existing stable install.

Supported inputs are common ABI-v1 Q4_0 weights and F32 M1 activations, for K4096/N11008 and K2048/N5632. The backend recognizes two adjacent matmuls with the same input, followed by SwiGLU consuming their separate outputs. Both outputs must have one graph consumer. It does not concatenate, duplicate, or repack weights.

One SDNN invocation uses the same completed activation window for both projections. The existing eight-owner DMA/DS/MAC/EW/RS/DMAOUT pipeline, inner-loop overlap, K32 scale correction, FP32 ordering, fences and projection-boundary drain are unchanged. One CL grid independently sums both partial arrays. A pair saves one SDNN launch, one sum launch, and one activation preload; activation quantization was already shared and is not counted as a new saving.

With activation sharing ON, exactly two lease consumers are released after success. With sharing OFF, the pair owns a local lease. Only a successful pair skips the second graph matmul. Rejection leaves both ordinary nodes executable. Plans do not cross graph execution boundaries.

Views, split/device-mismatched buffers, unsupported dtype/shape, nonadjacent graph partitions, extra output consumers, weight/input/output aliases, stale/foreign leases, incompatible MAC/DMA settings and insufficient workspace fall back. More input consumers are allowed and retain the lease. Output DMA alignment is 64 bytes. A rejected API call writes neither output.

L3 partial requests are 11,272,192 bytes (7B) or 2,883,584 bytes (Tiny), below the 16 MiB limit. Both arrays remain owned under the runtime lock through the completed CL sum. A failed L3 growth returns false and ordinary single-output GEMV may reserve its smaller workspace. No large weight cache is added.

The PP main body uses GEMM, but the last-layer output-row selection can produce an M1 FFN. Standard PP512 therefore executes one pair as well; PP must be checked in performance and quality tests.

## Build, install, load

Use separate build/install directories and the existing XRE 4.33 / XDNN / XTDK 2.7.98.2 environment:

```bash
cmake -S <library> -B <new-library-build> -DCMAKE_BUILD_TYPE=Release \
  -DKUNLUN_XRE=<existing-xre> -DKUNLUN_XDNN=<existing-xdnn> \
  -DKUNLUN_XTDK=<existing-xtdk> -DCMAKE_INSTALL_PREFIX=<new-install>
cmake --build <new-library-build> -j4
cmake --install <new-library-build>
cmake -S <llama> -B <new-llama-build> -DGGML_KUNLUN=ON \
  -DKUNLUN_USE_SYSTEM_LIBRARY=ON -DCMAKE_PREFIX_PATH=<new-install> \
  -DCMAKE_BUILD_TYPE=Release
cmake --build <new-llama-build> --target llama-bench -j4
```

For verified existing images, `-DKUNLUN_PREBUILT_KERNELS=<directory>` replaces source compilation. That directory must contain every required `.bin` and `.entry`, not just the pair files. Pair files are:

- `kunlun-common-gemv-pair-nmajor.bin/.entry`
- `kunlun-common-gemv-pair-stream.bin/.entry`
- `kunlun-common-sum-pair.bin/.entry`

CMake builds/installs them through the same kernel target as other images. `KUNLUN_KERNEL_DIR=<new-install>/lib/kunlunk200/kernels` selects the matching installed images. Set `LD_LIBRARY_PATH` to the new library and llama build directories before older locations. A missing image is a load error, not a numerical fallback. Keep the library/host API/backend and image SHA manifest together.

The existing `CommonQ4PairStatistics` ABI is retained. `common_q4_pair_last_status` adds a decline-status accessor without changing that struct. Its status describes the latest attempt under the workspace lock, not a process-wide history.

## Diagnostics and performance

`GGML_KUNLUN_Q4_GATE_UP_PAIR_STATS=1` prints aggregate enable-mode, successful-call, graph-ineligibility and API-decline reasons at backend destruction. It does not add device waits. Leave it unset/0 for normal measurements. Graph skip counts include observed ineligible adjacent matmuls and graphs without a pair; they are not attempted kernel calls. `workspace`, `alias`, `lease`, `configuration` and `weight_abi` are execution-decline categories.

The normal backend contains no layout-investigation L3 floor, code-load-order or allocation-trace API. Benchmark-only affinity and optional same-context/split timing belong to llama-bench, not inference. See `tools/llama-bench/BENCH_AFFINITY.ja.md`.

Compare OFF/ON with the same explicit main-thread CPU policy and other options. Fix `KUNLUN_Q4_GEMV_SUM_FAST=0` and `GGML_KUNLUN_SWIGLU_ABSMAX=0` when isolating this feature. Retain slow processes and use independent processes as the statistical unit. Performance is host/IRQ-dependent; the option does not move IRQs, change clocks or force CPU placement in normal inference.

## Development baseline

The 2026-11-03 Auto baseline is recorded in `results/gate-up-auto-20261103/final-state.json`, with actual source hashes, cumulative uncommitted patches, image/DSO/benchmark hashes and build/environment records. Git HEAD alone is insufficient. Existing opt-in results remain unchanged. See `docs/K200_BASELINE.ja.md` for defaults and the explicit benchmark profile.

## Optional controller address preparation

`KUNLUN_Q4_PAIR_PREADDR=1` enables an independent, default-OFF Pair-only image for the two supported shapes. DMA address calculation is done before the existing completion wait; actual command issue, all fences/barriers, shared buffers and FP32 order remain unchanged. Unset/0/invalid values use the original Pair. Pair default Auto is unchanged. See Q4_PAIR_PREADDR.ja.md and results/gate-up-critical-20261103/REPORT.ja.md for the measured benefit and limits.

## Optional shape constants

`KUNLUN_Q4_PAIR_DMA_CONST=1` enables default-OFF Pair-only shape specialization for K4096/N11008 and K2048/N5632 at M1. It composes with KUNLUN_Q4_PAIR_PREADDR without removing any DMA/config/fence/barrier. See Q4_PAIR_DMA_CONST.ja.md for build, fallback and measurement limits.
