# Indexed F16 cache updates

KunlunK200 0.4.1 extends the existing contiguous transposed-cache path to more than 32 tokens. The `SetRowsLayout` ABI and `set_rows_f16` signature remain unchanged. Previous libraries accept only `cache_tokens <= 32`; applications using the extended contract must require 0.4.1.

For `width=1`, `dim2=dim3=1`, contiguous source and indices, and `cache_features` divisible by 8, the mapping is:

```
indices[token*features+feature] = feature*cache_stride + start + token
dst[feature*cache_stride + start + token] = F16(src[token*features+feature])
```

`features*tokens` must equal `rows`, and `features*cache_stride` must equal `destination_rows`. Every index is validated on the device, including the upper word of I64 indices. The start and end must fit inside the cache. Host pointer alignment, byte extents, disjoint storage and layout validation remain mandatory.

The CL kernel reads eight features at a time and transposes at most 32 tokens in bounded local arrays. It writes one contiguous segment of up to 64 bytes per feature. A final partial tile writes only its actual tokens. No cache padding, unused slots, other heads or allocation guards are overwritten. F32 input uses the previous round-to-nearest-even F16 conversion; F16 input is copied bitwise. Tokens=1 and general strided layouts retain the existing paths.

The workspace remains 512 bytes per `Kernel` and owning PD. It stores the configuration and error status. There is no extra HBM or L3 activation workspace and no activation readback. The existing runtime mutex, kernel lifetime, configuration upload, completion wait and four-byte status readback remain in place. Invalid indices return false; callers must stop, because some valid tiles may already have completed.

The GGML integration recognizes the complete cache mapping before choosing this path. Duplicate or irregular indices use the established general/legacy behavior. The index readback and host mapping scan are retained; this change removes the large scalar-DMA cost rather than weakening index validation.

## Reproduction

```
source /home/unagidojyou/kunlun-k200/env.sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4 --target k200-set-rows-bench
# PD, width, rows, destination rows, source bytes, index bytes, gapped, invalid, features
build/k200-set-rows-bench 0 1 524288 4194304 4 8 0 0 4096
# Identical source and indices, general device scatter for comparison:
KUNLUN_SET_ROWS_GENERIC=1 build/k200-set-rows-bench 0 1 524288 4194304 4 8 0 0 4096
```

Use the existing thermal guard for all device execution. The benchmark resets the cache and checks allocation guards and every byte against CPU F16 conversion. Its five timing groups each contain 20 updates; it reports the minimum, median and maximum group mean. The `KUNLUN_SET_ROWS_GENERIC` variable affects this benchmark only.
