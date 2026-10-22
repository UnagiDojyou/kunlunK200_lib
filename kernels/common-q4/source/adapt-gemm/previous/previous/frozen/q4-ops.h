#pragma once
#include "frozen/frozen/previous/baseline/gemm-ops.h"
#include "frozen/frozen/previous/timing.h"
static inline __device__ void load_a(_global_ptr_ const int8_t* a,int m0,int k,int kp,int kt){
    _global_ptr_ const int8_t* p=gptr_i8(a,(long long)m0*kp+k);
    dma_cfg(DMA_L2_0|DMA_INT8,DMA_GM|DMA_INT8|highptr(p),0.f);dma_cfg_2d(32,kt,kp);
    dma_i_2d((int8_t*)0,lowptr(p),kt);
}
static inline __device__ void shuffle_a(int dst,int kt){
    ds_shuffle_cfg_datatype(DS_INT8);ds_shuffle_cfg_blkx(kt);ds_cfg_output_bank(NBANKS);
    v32i8* d=&((v32i8*)0)[dst];ds_shuffle_batch(&d[0][0],(int8_t*)0,kt,DS_L1D);ds_shuffle_batch(&d[0][1],(int8_t*)(16*kt),kt,DS_L1D);
}
static inline __device__ void load_w(_global_ptr_ const int8_t* p,int buf,int kt,int nt){
    dma_cfg(DMA_L2_1|DMA_INT8,DMA_GM|DMA_INT8|highptr(p),0.f);dma_i((int8_t*)(buf*(kt/2)*nt),lowptr(p),(kt/2)*nt);
}
static inline __device__ void shuffle_w(int buf,int kt,int nt,int dst){
    ds_shuffle_cfg_datatype(DS_INT16);ds_shuffle_cfg_blkx(kt/2);ds_cfg_output_bank(NBANKS);
    for(int ns=0;ns<nt/32;++ns){v32i8* w=&((v32i8*)0)[dst+ns*(kt/2)];ds_shuffle_batch(&w[0][0],(int8_t*)(buf*(kt/2)*nt+ns*(kt/2)*32),kt/2,DS_L1W);}
}
static inline __device__ void dot_w4(int a,int kt,int nt,int rows,int w,int acc){
    mm_cfg_stride(1);mm_cfg_dequant_scale(1.f);mm_cfg_basic_int8(kt,rows,rows);
    for(int ns=0;ns<nt/32;++ns){v16f32* e=&((v16f32*)0)[ns*rows*2];v32i8* d=&((v32i8*)0)[a];v32i8* b=&((v32i8*)0)[w+ns*(kt/2)];
        if(acc){mm_acc_int4(e,d,b,MAC_INT8_D8W4);}else{mm_int4(e,d,b,MAC_INT8_D8W4);}
    }
}
