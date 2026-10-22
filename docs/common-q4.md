# Common Q4_0 GEMV/GEMM (opt-in)

The backend can retain one versioned common-F Q4 weight region per tensor and owning PD, use GEMM for prompt matrices and GEMV for one-token matrices. Existing settings and defaults are preserved. This requires KunlunK200 >=0.4.0 (the pinned vendor submodule or an installed package).

## Settings and priority

| Mode | GGML_KUNLUN_COMMON_Q4 | GGML_KUNLUN_NATIVE_Q4 | GGML_KUNLUN_REQUANT_I8 |
|---|---|---|---|
| Common-F GEMV + GEMM | 1 | 0 | 1 |
| Legacy native GEMV / legacy CL matrices | 0 | 1 | 1 |
| CDNN INT8 | 0 | 0 | 1 |
| Original direct Q4 | 0 | 0 | 0 |
| CPU | 0 | 0 | 0; additionally -ngl 0 |

For eligible tensors: common > legacy native > requant I8. If common is unsupported, the remaining settings choose the previous storage/compute path. Requant I8 retains its existing nonzero-string semantics. COMMON_Q4 accepts exactly 1; invalid values are logged and disabled. Storage is selected during model allocation. Forward dispatch follows the stored handle, so changing environment variables after loading cannot reinterpret its bytes. Reload the model to switch modes.

KUNLUN_BROADCAST_VECTOR=1 is a separate library opt-in for contiguous row-broadcast ADD/MUL. It is not implied by COMMON_Q4 and does not alter defaults. Set it identically for CDNN and common-F in PP performance comparisons.

## Tensor and ownership contract

Common weights must be contiguous, non-view, 2D Q4_0 matrices, K divisible by 32, K<=16384, N<=32000. MUL_MAT uses W[N,K], X[M,K] and F32 Y[M,N]. F32 contiguous matrix activation is accepted by this GGML connection. F16 activations, noncontiguous activation views, weight views and batched higher-dimensional tensors follow existing backend placement/CPU fallback. Contiguous F32 activation views may run when matrix layout and element alignment hold. The standalone library also accepts finite F16 activation and partial K32.

The backend retains the original GGML Q4 host shadow for tensor reads, partial updates and CPU scheduling. A normal load packs/uploads once. A partial weight update deliberately rebuilds the affected resident PD region; it is not a forward operation. Ordinary tensors borrow a backend-owned pooled HBM region; Row Split has one region for each nonempty owning PD. Handle destruction precedes pool release. A region counter counts logical tensor suballocations, not xpu_malloc calls.

The typed library handle checks ABI version, quant_type=2, layout, scale order, padded dimensions, payload size, initialization and PD. Legacy pointers cannot be passed as common handles. GGML additionally checks K/N/type/device and the stored region pointer. There are no separate PP and TG weight copies. Workspaces belong to backend contexts/PDs; the synchronous library call and shared runtime mutex keep weights, scratch and shared SDK L3 live through completion.

## Dispatch

M=1 uses two-component A8 common-F GEMV. M>1 uses Row A8 SD GEMM, chunks <=128 and physical rows 8/32/64/128. Padding, cropping and one-row tails of matrix requests remain on XPU. Combined512/256, Wide, Rows64/128, Combo2/4 retain the tested kernel variants and L1D capacity checks; see vendor/KunlunK200/docs/common-q4.md.

Row Split pads each local N independently to 256, then crops before existing gather. It therefore handles local rows such as 5504 without sending an invalid N to GEMV. Single PD and Layer Split use the same API. Existing peer copies/gather and synchronization are preserved.

Known GEMV L3 capacity/OOM/not-supported conditions choose an exact common-layout CL implementation with a reason and counter. Other XPU errors abort immediately. The old native CL path is never used to decode common-F bytes.

## Diagnostics

COMMON_Q4 enables common totals at exit. GGML_KUNLUN_DIAGNOSTICS=1 also enables existing path totals. Logs show first use of each M/PD, GEMV/GEMM counts, exact fallback, per-PD payload/regions/uploads and CPU pack/H2D time. Optional K200_PROFILE_PHASE tags phase counters. Smoke tests set pp/tg/warmup/cleanup; library clients need not set it. Workspace logs distinguish HBM scratch, shared L3 capacity and Combo slice scratch.

Successful PP-to-TG validation requires uploads==regions and upload_bytes==payload_bytes after a normal single model load, native_cl=0 and exact_cl=0 for supported main shapes. Report the PP M=1 work left by final-logits graph pruning separately from M>1 GEMM. Counters are cumulative for a process, so model reload tests legitimately increment regions and uploads again.

## Build and run

~~~bash
source /path/to/workspace/env.sh
cmake -S . -B build-k200 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_KUNLUN=ON -DKUNLUN_USE_SYSTEM_LIBRARY=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build-k200 -j4 --target llama-app llama-bench kunlun-llm-smoke kunlun-gemv-bench

export GGML_KUNLUN_COMMON_Q4=1 GGML_KUNLUN_NATIVE_Q4=0 GGML_KUNLUN_REQUANT_I8=1
export KUNLUN_BROADCAST_VECTOR=1 GGML_KUNLUN_DIAGNOSTICS=1
# Wrap every hardware invocation with the existing thermal guard and PD fail-stop monitor.
build-k200/bin/llama completion -m /models/model-q4_0.gguf \
  -ngl 99 -dev KUNLUN0 -sm none -c 2048 -b 128 -ub 128 -t 6 \
  -fa off -n 32 --temp 0 --seed 1 --ignore-eos --simple-io -no-cnv \
  -p "Explain why the sky is blue."
# Layer / Row: -dev KUNLUN0,KUNLUN1 -sm layer|row -ts 1,1
# CPU: -ngl 0 (and disable COMMON_Q4/NATIVE_Q4/REQUANT_I8 as in the table).
~~~

For an installed package, use KUNLUN_USE_SYSTEM_LIBRARY=ON, CMAKE_PREFIX_PATH pointing to its prefix, GGML_BACKEND_DL=ON and GGML_NATIVE=OFF. GGML_KUNLUN=OFF requires no SDK discovery.

## Numerical and performance limits

Common-F preserves original W4 values and FP16 block scales. GEMM Row A8 changes activation values; CPU-logit or CDNN quality equivalence is not claimed. The historical 7B Row A8 PPL 6.0438 to 6.1421 is a prior CPU experiment, not a PPL measurement of this integrated mixed PP/TG path. Current evidence separates implementation error from quantization error and measures teacher-forced logits/KL plus independent generation.

Formal legacy compact-2d, standalone dual16 and common-F are different baselines. Common-F GEMV has a known up-to-about-6% SD-body regression against standalone dual16; full LLM TG can differ more because launch, synchronization, CPU attention and F16 KV transfers remain. Common Q4 forward does not read activation back for quantization; other model operations still transfer data. See the Japanese integration report and its raw timing/memory/counter tables before choosing a mode. No default or remote branch is updated by this integration.
