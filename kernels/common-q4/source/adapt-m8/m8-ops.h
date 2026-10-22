#pragma once
#define q40_w_ds original_weight_ds
#include "base-ops.h"
#undef q40_w_ds
// Concatenating N32 strips within each of 16 source banks lets one DS batch
// populate exactly the same L1W rows as the original per-strip batches.
static inline __device__ void q40_w_ds(int buf,int kt,int nt){
    if(!FUSE_BANKS){original_weight_ds(buf,kt,nt);return;}
    int length=(kt/2)*(nt/32);
    ds_shuffle_cfg_datatype(DS_INT16);ds_shuffle_cfg_blkx(length);ds_cfg_output_bank(NBANKS);
    ds_shuffle_batch(&((v32i8*)0)[buf*length][0],(int8_t*)(buf*(kt/2)*nt),length,DS_L1W);
}
static inline __device__ void preload_a(_global_ptr_ const int8_t* a,int offset,int kp,int width,int init){
    _global_ptr_ const int8_t* p=gptr_i8(a,offset);
    dma_cfg(DMA_L2_0|DMA_INT8,DMA_GM|DMA_INT8|highptr(p),0.f);
    // First fill initializes all 32 shuffled rows. Later equal-width fills
    // update only eight rows; the remaining 24 rows retain verified zeros.
    dma_cfg_2d(init?32:8,width,kp);dma_i_2d((int8_t*)0,lowptr(p),width);
}
static inline __device__ void q40_reduce(int pslot,int sslot,int blocks,int rows,int nt,int acc,int first,unsigned int* dt){
    if(!DIRECT_FIRST){original_bulk_reduce(pslot,sslot,blocks,rows,nt,acc,first,dt);return;}
    int plane=rows*(nt/16);v16f32* e=(v16f32*)0;int p=pslot*blocks*plane;
    DETAIL(13,{
        if(first){ew_cfg_activation_type(EW_NOACT);ew_cfg_findmax(0,0.f);ew_cfg_col_size(NBANKS);ew_cfg_stream_size(blocks*plane);}
        ew_cfg_coeff_scalar(0.f,0.f,1.f);
        ew_dsmul(&e[first?acc:p],&e[p],&e[2*blocks*plane+sslot*blocks*plane],EW_L1E);
    });COUNT(25);
    // The first group can initialize acc directly. Later groups keep the
    // multiplication dependency fence and preserve each original K32 dW.
    if(first)return;
    DETAIL(14,xfence());COUNT(26);
    DETAIL(13,{ew_cfg_coeff_scalar(1.f,1.f,0.f);ew_dsmadd(&e[acc],&e[p],&e[acc],EW_L1E);});COUNT(25);
}
