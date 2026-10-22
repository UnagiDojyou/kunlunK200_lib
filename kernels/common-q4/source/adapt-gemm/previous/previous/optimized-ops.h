#pragma once
#define q40_s_dma previous_scale_dma
#define q40_s_ds previous_scale_ds
#define q40_reduce previous_reduce
#include "frozen/q40-lanes-ops.h"
#undef q40_s_dma
#undef q40_s_ds
#undef q40_reduce
#include "detail.h"
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
