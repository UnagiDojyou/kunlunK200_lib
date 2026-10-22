#pragma once
#include "q4-ops.h"
static inline __device__ _global_ptr_ const unsigned short* hptr(_global_ptr_ const unsigned short* p,long long n){return (_global_ptr_ const unsigned short*)((long long)p+2*n);}
static inline __device__ void q40_w_dma(_global_ptr_ const int8_t* p,int buf,int kt,int nt){
    dma_cfg(DMA_L2_0|DMA_INT8,DMA_GM|DMA_INT8|highptr(p),0.f);dma_i((int8_t*)(buf*(kt/2)*nt),lowptr(p),(kt/2)*nt);
}
static inline __device__ void q40_w_ds(int buf,int kt,int nt){
    ds_shuffle_cfg_datatype(DS_INT16);ds_shuffle_cfg_blkx(kt/2);ds_cfg_output_bank(NBANKS);
    for(int ns=0;ns<nt/32;++ns)ds_shuffle_batch(&((v32i8*)0)[(buf*(nt/32)+ns)*(kt/2)][0],(int8_t*)(buf*(kt/2)*nt+ns*(kt/2)*32),kt/2,DS_L1W);
}
static inline __device__ void q40_s_dma(_global_ptr_ const unsigned short* p,int buf,int blocks,int rows,int nt){
    dma_cfg(DMA_L2_1|DMA_FP32,DMA_GM|DMA_FP16|highptr(p),0.f);dma_cfg_2d(rows,nt*4,0);
    for(int b=0;b<blocks;++b)dma_i_2d((float*)((buf*blocks+b)*rows*nt*4),lowptr(hptr(p,b*nt)),nt*2);
}
static inline __device__ void q40_s_ds(int buf,int blocks,int rows,int nt,int slot){
    int plane=rows*(nt/16);ds_shuffle_cfg_datatype(DS_FP32);ds_shuffle_cfg_blkx(nt);ds_cfg_output_bank(NBANKS);
    for(int b=0;b<blocks;++b)for(int ns=0;ns<nt/16;++ns){
        v16f32* dst=&((v16f32*)0)[2*blocks*plane+(slot*blocks+b)*plane+ns*rows];
        ds_shuffle_coa(rows,&dst[0][0],(float*)(((buf*blocks+b)*rows*nt+ns*16)*4),DS_L1E);
    }
}
static inline __device__ void q40_row_scale_dma(_global_ptr_ const float* p,int rows){
    dma_cfg(DMA_L2_1|DMA_FP32,DMA_GM|DMA_FP32|highptr(p),0.f);dma_i((float*)0,lowptr(p),rows*64);
}
static inline __device__ void q40_row_scale_ds(int rows,int nt,int dst){
    ds_shuffle_cfg_datatype(DS_FP32);ds_shuffle_cfg_blkx(16);ds_cfg_output_bank(NBANKS);
    for(int ns=0;ns<nt/16;++ns)ds_shuffle_coa(rows,&((v16f32*)0)[dst+ns*rows][0],(float*)0,DS_L1E);
}
static inline __device__ void q40_mac(int a,int buf,int kt,int nt,int rows,int pslot){
    int blocks=kt/32,plane=rows*(nt/16);mm_cfg_stride(1);mm_cfg_dequant_scale(1.f);mm_cfg_basic_int8(32,rows,rows);
    for(int b=0;b<blocks;++b)for(int ns=0;ns<nt/32;++ns){
        mm_int4(&((v16f32*)0)[(pslot*blocks+b)*plane+ns*rows*2],&((v32i8*)0)[a+b*32],&((v32i8*)0)[(buf*(nt/32)+ns)*(kt/2)+b*16],MAC_INT8_D8W4);
    }
}
static inline __device__ void q40_reduce(int pslot,int sslot,int blocks,int rows,int nt,int acc,int first){
    int plane=rows*(nt/16);v16f32* e=(v16f32*)0;v16f32* temp=&e[5*blocks*plane];
    ew_cfg_activation_type(EW_NOACT);ew_cfg_findmax(0,0.f);ew_cfg_col_size(NBANKS);ew_cfg_stream_size(blocks*plane);ew_cfg_coeff_scalar(0.f,0.f,1.f);
    ew_dsmul(temp,&e[pslot*blocks*plane],&e[2*blocks*plane+sslot*blocks*plane],EW_L1E);xfence();
    ew_cfg_stream_size(blocks*plane);ew_cfg_coeff_scalar(1.f,first?0.f:1.f,0.f);
    ew_dsmadd(&e[acc],temp,first?temp:&e[acc],EW_L1E);
}
static inline __device__ void q40_finish(int rows,int nt,int acc,int scale,int blocks){
    int plane=rows*(nt/16);v16f32* e=(v16f32*)0;
    ew_cfg_activation_type(EW_NOACT);ew_cfg_findmax(0,0.f);ew_cfg_col_size(NBANKS);ew_cfg_coeff_scalar(1.f,1.f,0.f);
    for(int live=blocks;live>1;live/=2){ew_cfg_stream_size(live/2*plane);ew_dsmadd(&e[acc],&e[acc],&e[acc+live/2*plane],EW_L1E);xfence();}
    ew_cfg_stream_size(plane);ew_cfg_coeff_scalar(0.f,0.f,1.f);
    ew_dsmul((v16f32*)0,&((v16f32*)0)[acc],&((v16f32*)0)[scale],EW_L2E);
}
static inline __device__ void q40_rs(int physical,int valid,int nt){
    rs_row_cfg_stride(1,nt/16);rs_row_cfg_loop(valid);
    for(int ns=0;ns<nt/16;++ns)rs_row_batch(&((v16f32*)0)[ns][0],&((v16f32*)0)[ns*physical][0],NBANKS);
}
