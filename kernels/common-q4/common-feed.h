#pragma once
#ifndef FUSE_BANKS
#define FUSE_BANKS 0
#endif
// All 2D byte strides use the documented dma_cfg_2d interface already used
// by the frozen A/scale feeders. Only weight addresses and feed order change.
static inline __device__ void common_w_dma(_global_ptr_ const int8_t* w,int n0,int k0,int kp,int buf,int kt,int nt){
    const int ns=nt/32,base=buf*(kt/2)*nt;
#if COMMON_LAYOUT==2
    int macro=k0/512*512,span=kp-macro;if(span>512)span=512;
    for(int s=0;s<ns;++s){
        _global_ptr_ const int8_t* src=gptr_i8(w,((long long)(n0/32+s)*(kp/32)+macro/32)*512+k0-macro);
        dma_cfg(DMA_L2_0|DMA_INT8,DMA_GM|DMA_INT8|highptr(src),0.f);
        dma_cfg_2d(16,FUSE_BANKS?ns*kt:kt,span);
        dma_i_2d((int8_t*)(base+(FUSE_BANKS?s*kt:s*kt*16)),lowptr(src),kt);
    }
#else
    _global_ptr_ const int8_t* src=gptr_i8(w,((long long)(n0/32)*(kp/32)+k0/32)*512);
    dma_cfg(DMA_L2_0|DMA_INT8,DMA_GM|DMA_INT8|highptr(src),0.f);
    dma_cfg_2d(ns,kt*16,kp*16);dma_i_2d((int8_t*)base,lowptr(src),kt*16);
#endif
}
static inline __device__ void common_w_ds(int buf,int kt,int nt){
    int ns=nt/32;ds_shuffle_cfg_datatype(DS_INT16);ds_cfg_output_bank(NBANKS);
#if COMMON_LAYOUT==2
    if(FUSE_BANKS){int length=kt/2*ns;ds_shuffle_cfg_blkx(length);ds_shuffle_batch(&((v32i8*)0)[buf*length][0],(int8_t*)(buf*kt/2*nt),length,DS_L1W);}
    else{ds_shuffle_cfg_blkx(kt/2);for(int s=0;s<ns;++s)ds_shuffle_batch(&((v32i8*)0)[(buf*ns+s)*(kt/2)][0],(int8_t*)(buf*kt/2*nt+s*kt*16),kt/2,DS_L1W);}
#elif COMMON_LAYOUT==0
    ds_shuffle_cfg_blkx(16);for(int s=0;s<ns;++s)ds_shuffle_coa(kt/2,&((v32i8*)0)[(buf*ns+s)*(kt/2)][0],(int8_t*)(buf*kt/2*nt+s*kt*16),DS_L1W);
#else
    ds_shuffle_cfg_blkx(16);for(int s=0;s<ns;++s)for(int b=0;b<kt/32;++b)ds_shuffle_batch(&((v32i8*)0)[(buf*ns+s)*(kt/2)+b*16][0],(int8_t*)(buf*kt/2*nt+s*kt*16+b*512),16,DS_L1W);
#endif
}
static inline __device__ void common_s_dma(_global_ptr_ const unsigned short* scales,int n0,int kb,int kp,int np,int buf,int blocks,int nt){
    const int base=buf*blocks*nt*4;
#if COMMON_SCALE==1
    _global_ptr_ const unsigned short* src=hptr(scales,(long long)kb*np+n0);
    dma_cfg(DMA_L2_1|DMA_FP32,DMA_GM|DMA_FP16|highptr(src),0.f);dma_cfg_2d(blocks,nt*4,np*2);dma_i_2d((float*)base,lowptr(src),nt*2);
#elif COMMON_SCALE==2
    _global_ptr_ const unsigned short* src=hptr(scales,((long long)(n0/32)*(kp/32)+kb)*32);
    dma_cfg(DMA_L2_1|DMA_FP32,DMA_GM|DMA_FP16|highptr(src),0.f);dma_cfg_2d(nt/32,blocks*128,(kp/32)*64);dma_i_2d((float*)base,lowptr(src),blocks*64);
#else
    for(int s=0;s<nt/32;++s){
        _global_ptr_ const unsigned short* src=hptr(scales,((long long)(n0/32+s)*(kp/32)+kb)*32);
        dma_cfg(DMA_L2_1|DMA_FP32,DMA_GM|DMA_FP16|highptr(src),0.f);dma_cfg_2d(blocks,nt*4,64);dma_i_2d((float*)(base+s*128),lowptr(src),64);
    }
#endif
}
static inline __device__ void common_s_ds(int buf,int blocks,int rows,int nt,int slot){
#if COMMON_SCALE==2
    int plane=rows*(nt/16);v16f32* e=(v16f32*)0;
    ds_shuffle_cfg_datatype(DS_FP32);ds_shuffle_cfg_blkx(0);ds_cfg_output_bank(NBANKS);
    for(int b=0;b<blocks;++b)for(int ns=0;ns<nt/16;++ns){
#if defined(SCALE_COMPACT) && SCALE_COMPACT
        int dest=2*blocks*plane+slot*blocks*(nt/16)+b*(nt/16)+ns;int repeat=1;
#else
        int dest=2*blocks*plane+(slot*blocks+b)*plane+ns*rows;int repeat=rows;
#endif
        ds_shuffle_coa(repeat,&e[dest][0],(float*)(buf*blocks*nt*4+((ns/2)*blocks+b)*128+(ns%2)*64),DS_L1E);
    }
#else
    q40_s_ds(buf,blocks,rows,nt,slot);
#endif
}
