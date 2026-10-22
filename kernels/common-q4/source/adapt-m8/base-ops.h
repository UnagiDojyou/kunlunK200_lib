#pragma once
#define q40_s_dma previous_scale_dma
#define q40_s_ds previous_scale_ds
#define q40_reduce previous_reduce
#include "previous/previous/frozen/q40-lanes-ops.h"
#undef q40_s_dma
#undef q40_s_ds
#undef q40_reduce
#include "previous/previous/detail.h"
// Compact original FP16 scales are converted once; DS repeats one vector to M.
static inline __device__ void q40_s_dma(_global_ptr_ const unsigned short* p,int buf,int blocks,int rows,int nt){
    dma_cfg(DMA_L2_1|DMA_FP32,DMA_GM|DMA_FP16|highptr(p),0.f);
    dma_i((float*)(buf*blocks*nt*4),lowptr(p),blocks*nt*2);
}
static inline __device__ void q40_s_ds(int buf,int blocks,int rows,int nt,int slot){
    int plane=rows*(nt/16);v16f32* e=(v16f32*)0;
    ds_shuffle_cfg_datatype(DS_FP32);ds_shuffle_cfg_blkx(0);ds_cfg_output_bank(NBANKS);
    for(int b=0;b<blocks;++b)for(int ns=0;ns<nt/16;++ns)
        ds_shuffle_coa(rows,&e[2*blocks*plane+(slot*blocks+b)*plane+ns*rows][0],(float*)(((buf*blocks+b)*nt+ns*16)*4),DS_L1E);
}
// All K32 scales remain distinct. Only after multiplying each partial by its
// original dW may adjacent lane halves be added. No partial is sent to HBM.
static inline __device__ void original_bulk_reduce(int pslot,int sslot,int blocks,int rows,int nt,int acc,int first,unsigned int* dt){
    int plane=rows*(nt/16);v16f32* e=(v16f32*)0;int p=pslot*blocks*plane;
    DETAIL(13,{
        ew_cfg_activation_type(EW_NOACT);ew_cfg_findmax(0,0.f);ew_cfg_col_size(NBANKS);ew_cfg_stream_size(blocks*plane);
        ew_cfg_coeff_scalar(0.f,0.f,1.f);ew_dsmul(&e[p],&e[p],&e[2*blocks*plane+sslot*blocks*plane],EW_L1E);
    });COUNT(25);DETAIL(14,xfence());COUNT(26);
#if FOLD_LANES
    DETAIL(13,{
        ew_cfg_stream_size(blocks/2*plane);ew_cfg_coeff_scalar(1.f,1.f,0.f);
        ew_dsmadd(&e[p],&e[p],&e[p+blocks/2*plane],EW_L1E);
    });COUNT(25);DETAIL(14,xfence());COUNT(26);
    blocks/=2;
#endif
    DETAIL(13,{
        ew_cfg_stream_size(blocks*plane);ew_cfg_coeff_scalar(1.f,first?0.f:1.f,0.f);
        ew_dsmadd(&e[acc],&e[p],&e[first?p:acc],EW_L1E);
    });COUNT(25);
}
