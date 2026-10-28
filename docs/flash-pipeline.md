# SDNN/CL FlashAttention pipeline (0.7.5)

`KUNLUN_FLASH_PIPELINE=off/auto/on` selects sequential, measured automatic, or experimental forced scheduling. An unset value selects `auto`; unknown values select sequential scheduling. Flash-Decoding dispatch still runs first.

Auto is limited to M1, H32, batch 1, block 1024 and query tile 128, with either D128/HK32 and 1024 < S <= 4352, or D64/HK4 and 8192 <= S <= 8448. It requires the existing reuse1024, ordered-online and wide-PV defaults and L3 workspace. PP stays sequential because repeated LLM measurements did not show a consistent win on both PDs. Other shapes remain available with `on` for comparison.

The first KQ follows preprocessing in the default stream. For each nonfinal block, online softmax runs in the CL stream while the next KQ runs in the SDNN stream. Joining CL makes P ready; PV then follows that KQ in the SDNN queue. Joining SDNN makes both the next score and the previous partial ready. The final online/PV/normalization chain shares one stream. Each API returns only after both owned streams are drained. MAC arithmetic, quantization and ordered FP32 accumulation are unchanged.

The tested XRE event API returns -807 (unknown ioctl) with this driver. This implementation uses stream completion joins, not events or device polling. A warm N-block call has 3*N+1 attention launches and 2*N necessary completion joins, excluding unchanged preprocessing. Diagnostics are separate from those joins.

`KUNLUN_FLASH_PIPELINE_BUFFER=auto/hbm/l3/state-hbm` controls experiments. Auto doubles only Score/P in L3 when it fits 16 MiB. Otherwise forced pipeline keeps the original workspace in L3 and places the second Score/P slot in HBM. `l3` falls back to sequential execution if both slots do not fit. `state-hbm` keeps both Score/P slots in L3 and moves partial/acc/stats to HBM. It was slower than the default placement and is not auto-selected. `KUNLUN_FLASH_HBM=1` is also excluded from automatic pipeline selection. Buffers retain their peak allocation until their Kernel owner is destroyed.

`KUNLUN_FLASH_PIPELINE_PROFILE=1` with `on` and the default optimized kernels records device-clock intervals. It adds timestamp stores, D2H and printing; its API time is diagnostic only. `KUNLUN_FLASH_TRACE` and existing stage profiling deliberately select sequential scheduling. `KUNLUN_FLASH_PIPELINE_STATS=1` prints activation counters at teardown without stage synchronization.

Build the normal CMake targets, then run under the temperature guard:

```sh
KUNLUN_FLASH_DECODING=off KUNLUN_FLASH_PIPELINE=on \
  build/k200-flash-pipeline-bench 0 128 2048 128 32 32
KUNLUN_FLASH_DECODING=off \
  build/k200-flash-pipeline-perf 0 128 2048 128 32 32
KUNLUN_FLASH_DECODING=off \
  build/k200-flash-pipeline-race 0 128 2048 128 32 32
```

The perf target warms one API owner, alternates off/on in ABBA order, and records 24 calls per mode. The race target changes Q/K/V and masks repeatedly and requires exact outputs. The general benchmark checks CPU FP64, input immutability, output guards and the sequential reference. See [the 0.7.6 publication guide](flash-kv-dma.md) for measured dispatch, comparison conditions and limitations. Full machine-specific evaluation archives remain separate from source.
