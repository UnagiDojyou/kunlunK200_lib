// Derived from sam6321/KunlunK200 5ead5bb (MIT, reference/LICENSE).
#pragma once
#include "reference/cdnn_ref.h"

// Two independent DMA/DS paths: even core L2_0 for A, odd core L2_1 for W.
// Conservative allocation: each path, including both slots, fits 32 KiB.
static inline __device__ void stage_dma(_global_ptr_ const int8_t* p, int buf,
                                       int rows, int cols, int ld, int weight) {
    int8_t* dst = (int8_t*)(buf * rows * cols);
    dma_cfg((weight ? DMA_L2_1 : DMA_L2_0) | DMA_INT8,
            DMA_GM | DMA_INT8 | highptr(p), 0.f);
    if (cols == ld) { dma_i(dst, lowptr(p), rows * cols); }
    else {
        dma_cfg_2d(rows, cols, ld);
        dma_i_2d(dst, lowptr(p), cols);
    }
}

static inline __device__ void stage_ds(int buf, int pk, int nt, int weight, int aoff=-1) {
    int8_t* src = (int8_t*)(buf * (weight ? pk * nt : 32 * pk));
    int offset=(!weight && aoff>=0) ? aoff : buf * (weight ? pk * (nt / 32) : pk);
    v32i8* dst = &((v32i8*)0)[offset];
    ds_shuffle_cfg_datatype(DS_INT8);
    ds_shuffle_cfg_blkx(weight ? nt : pk);
    ds_cfg_output_bank(NBANKS);
    if (weight) {
        for (int n = 0; n < nt; n += 32) {
            ds_shuffle_coa(pk, &dst[(n / 32) * pk][0], src + n, DS_L1W);
            ds_shuffle_coa(pk, &dst[(n / 32) * pk][1], src + n + 16, DS_L1W);
        }
    } else {
        ds_shuffle_batch(&dst[0][0], src, pk, DS_L1D);
        ds_shuffle_batch(&dst[0][1], src + 16 * pk, pk, DS_L1D);
    }
}

static inline __device__ void stage_mac(int buf, int kt, int nt, int rows, int acc, int aoff=-1) {
    int pk = kt / 2;
    v32i8* d = &((v32i8*)0)[aoff>=0 ? aoff : buf * pk];
    v32i8* w = &((v32i8*)0)[buf * pk * (nt / 32)];
    v16f32* e = (v16f32*)0;
    mm_cfg_stride(1);
    mm_cfg_dequant_scale(1.f);
    mm_cfg_basic_int8(kt, rows, rows);
    for (int n = 0; n < nt / 32; ++n) {
        if (acc) { mm_acc_int4(&e[n * rows * 2], d, &w[n * pk], MAC_INT8_D4W4); }
        else { mm_int4(&e[n * rows * 2], d, &w[n * pk], MAC_INT8_D4W4); }
    }
}

static inline __device__ void stage_ew(int rows, int nt) {
    ew_cfg_activation_type(EW_NOACT);
    ew_cfg_coeff_scalar(1.f, 0.f, 0.f);
    ew_cfg_findmax(0, 0.f);
    ew_cfg_stream_size(rows * (nt / 16));
    ew_cfg_col_size(NBANKS);
    ew_dsmadd((v16f32*)0, (v16f32*)0, (v16f32*)0, EW_L2E);
}

static inline __device__ void stage_rs(int rows, int nt) {
    v16f32* e = (v16f32*)0;
    v16f32* r = (v16f32*)0;
    rs_row_cfg_stride(1, nt / 16);
    rs_row_cfg_loop(rows);
    for (int n = 0; n < nt / 16; ++n) {
        rs_row_batch(&r[n][0], &e[n * rows][0], NBANKS);
    }
}

static inline __device__ void stage_out(_global_ptr_ float* p, int rows, int nt, int ld) {
    dma_cfg(DMA_GM | DMA_FP32 | highptr(p), DMA_L2R | DMA_FP32, 0.f);
    dma_cfg_2d(rows, ld * 4, nt * 4);
    dma_o_2d(lowptr(p), (v16f32*)0, nt * 4);
}
