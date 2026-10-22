#pragma once
#define q40_s_dma previous_scale_dma
#define q40_s_ds previous_scale_ds
#define q40_reduce previous_reduce
#define q40_mac previous_mac
#include "previous/previous/frozen/q40-lanes-ops.h"
#undef q40_s_dma
#undef q40_s_ds
#undef q40_reduce
#undef q40_mac
#include "previous/previous/detail.h"
static inline __device__ void q40_s_dma(_global_ptr_ const unsigned short* p,int buf,int blocks,int rows,int nt){
    if(!SCALE_COMPACT){previous_scale_dma(p,buf,blocks,rows,nt);return;}
    dma_cfg(DMA_L2_1|DMA_FP32,DMA_GM|DMA_FP16|highptr(p),0.f);
    dma_i((float*)(buf*blocks*nt*4),lowptr(p),blocks*nt*2);
}
static inline __device__ void q40_s_ds(int buf,int blocks,int rows,int nt,int slot){
    if(!SCALE_COMPACT){previous_scale_ds(buf,blocks,rows,nt,slot);return;}
    int plane=rows*(nt/16),vectors=blocks*(nt/16);v16f32* e=(v16f32*)0;
    ds_shuffle_cfg_datatype(DS_FP32);ds_shuffle_cfg_blkx(nt);ds_cfg_output_bank(NBANKS);
    for(int b=0;b<blocks;++b)for(int ns=0;ns<nt/16;++ns)
        ds_shuffle_coa(1,&e[2*blocks*plane+slot*vectors+b*(nt/16)+ns][0],(float*)(((buf*blocks+b)*nt+ns*16)*4),DS_L1E);
}
static inline __device__ void q40_reduce(int pslot,int sslot,int blocks,int rows,int nt,int acc,int first,unsigned int* dt){
    int plane=rows*(nt/16),scale_slot=SCALE_COMPACT?blocks*(nt/16):blocks*plane;
    v16f32* e=(v16f32*)0;int temp=2*blocks*plane+3*scale_slot;
    DETAIL(13,{
        ew_cfg_activation_type(EW_NOACT);ew_cfg_findmax(0,0.f);ew_cfg_col_size(NBANKS);ew_cfg_stream_size(rows);
        for(int b=0;b<blocks;++b)for(int ns=0;ns<nt/16;++ns){
            int off=b*plane+ns*rows;int coff=SCALE_COMPACT?b*(nt/16)+ns:off;
            ew_cfg_coeff(&e[2*blocks*plane+sslot*scale_slot+coff],EW_FUSED&&!first?1.f:0.f,0.f,EW_COEFF0_VECTOR);
            ew_dsmadd(&e[(EW_FUSED?acc:temp)+off],&e[pslot*blocks*plane+off],&e[EW_FUSED&&!first?acc+off:pslot*blocks*plane+off],EW_L1E);
            COUNT(25);
        }
    });
    if(!EW_FUSED){
        DETAIL(14,xfence());COUNT(26);
        DETAIL(13,{
            ew_cfg_stream_size(blocks*plane);ew_cfg_coeff_scalar(1.f,first?0.f:1.f,0.f);
            ew_dsmadd(&e[acc],&e[temp],&e[first?temp:acc],EW_L1E);COUNT(25);
        });
    }
}

static inline __device__ void big_load_a(_global_ptr_ const int8_t* a,int m0,int k0,int kp,int groups){
    _global_ptr_ const int8_t* p=gptr_i8(a,(long long)m0*kp+k0);
    dma_cfg(DMA_L2_0|DMA_INT8,DMA_GM|DMA_INT8|highptr(p),0.f);dma_cfg_2d(groups*32,32,kp);
    dma_i_2d((int8_t*)0,lowptr(p),32);
}
static inline __device__ void big_shuffle_a(int dst,int groups){
    ds_shuffle_cfg_datatype(DS_INT8);ds_shuffle_cfg_blkx(32);ds_cfg_output_bank(16);
    for(int t=0;t<groups;++t){
        ds_shuffle_batch(&((v32i8*)0)[dst+t*32][0],(int8_t*)(t*1024),32,DS_L1D);
        ds_shuffle_batch(&((v32i8*)0)[dst+t*32][1],(int8_t*)(t*1024+512),32,DS_L1D);
    }
}
static inline __device__ void q40_mac(int a,int buf,int kt,int nt,int rows,int pslot){
    int blocks=kt/32,plane=rows*(nt/16),groups=(rows+31)/32;
    mm_cfg_stride(1);mm_cfg_dequant_scale(1.f);mm_cfg_basic_int8(32,rows,rows);
    for(int b=0;b<blocks;++b)for(int ns=0;ns<nt/32;++ns)
        mm_int4(&((v16f32*)0)[(pslot*blocks+b)*plane+ns*rows*2],&((v32i8*)0)[a+b*32*groups],&((v32i8*)0)[(buf*(nt/32)+ns)*(kt/2)+b*16],MAC_INT8_D8W4);
}
