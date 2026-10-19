// Compact KL1 CDNN helpers, copied from XDNN 2.0 gemm_libs_impl.h / findmax.h
// so we own a launchable reference without pulling the whole library TU.
// SRAM banks are separate address spaces; pointers are offsets from 0.
//
// FOOTGUN: xfence_unlock(dev) in sdcdnn_header.h expands to two statements
// ( __builtin_xpu_xfence(); __builtin_xpu_unlock(dev); ) with no do{}while(0)
// wrapper. Never use it as the body of a brace-less `if`/`for`: the unlock
// leaks out and runs unconditionally, unlocking an unlocked/invalid engine
// (mydev==-1 on non-engine cores) which 714s the card. Always brace it.
#pragma once
#include "xpu/kernel/sdcdnn_header.h"
#include "xpu/kernel/math.h"

static inline __device__ void cdnn_only_core0() {
    if (core_id() != 0) {
        return;
    }
}

// Device ALU is 32-bit. m * packed_K overflows signed int past 2 GB.
// long long is emulated; highptr already reads the top half of a global pointer.
static inline __device__ _global_ptr_ const int8_t* gptr_i8(
    _global_ptr_ const int8_t* p, long long byte_off) {
    return (_global_ptr_ const int8_t*)((long long)p + byte_off);
}

static inline __device__ _global_ptr_ float* gptr_f32(_global_ptr_ float* p,
                                                     long long elem_off) {
    return (_global_ptr_ float*)((long long)p + elem_off * (int)sizeof(float));
}

static inline __device__ _global_ptr_ const float* gptr_f32c(
    _global_ptr_ const float* p, long long elem_off) {
    return (_global_ptr_ const float*)((long long)p + elem_off * (int)sizeof(float));
}

static inline __device__ _global_ptr_ const int16_t* gptr_i16(
    _global_ptr_ const int16_t* p, long long elem_off) {
    return (_global_ptr_ const int16_t*)((long long)p + elem_off * (int)sizeof(int16_t));
}

static inline __device__ void dma_in_1d(void* l2, _global_ptr_ const void* gm,
                                        int nbytes, int dst_pos, int dtype) {
    xfence_lock(DMAIN_0);
    dma_cfg((dst_pos | dtype), (DMA_GM | dtype | highptr(gm)), 0.0f);
    dma_i(l2, lowptr(gm), nbytes);
    xfence_unlock(DMAIN_0);
}
