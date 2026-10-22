static inline __device__ void scale_dma(_global_ptr_ const unsigned short *s,int buf,int count,int blocks) {
    for(int h=0;h<2;++h){
        _global_ptr_ const unsigned short *p=(_global_ptr_ const unsigned short *)gptr_i16((_global_ptr_ const int16_t *)s,(long long)h*blocks*32);
        dma_cfg(DMA_L2_1|DMA_FP32,DMA_GM|DMA_FP16|highptr(p),0.f);
        dma_i((float *)(16384+buf*4096+h*2048),lowptr(p),count*64);
    }
}

static inline __device__ void commonv_s_dma(_global_ptr_ const unsigned short* scales,int n0,int kb,int k,int n,int buf,int count,int item){
#if GEMV_CACHE128
    if(item%2==0){
        _global_ptr_ const unsigned short* p=(_global_ptr_ const unsigned short*)gptr_i16((_global_ptr_ const int16_t*)scales,(long long)kb*n+n0);
        dma_cfg(DMA_L2_1|DMA_FP32,DMA_GM|DMA_FP16|highptr(p),0.f);dma_cfg_2d(count,512,n*2);dma_i_2d((float*)(16384+((item/2)&1)*8192),lowptr(p),256);
    }
#elif COMMON_SCALE==0
    scale_dma((_global_ptr_ const unsigned short*)gptr_i16((_global_ptr_ const int16_t*)scales,((long long)(n0/32)*(k/32)+kb)*32),buf,count,k/32);
#else
    for(int h=0;h<2;++h){
        _global_ptr_ const unsigned short* p=(_global_ptr_ const unsigned short*)gptr_i16((_global_ptr_ const int16_t*)scales,(long long)kb*n+n0+h*32);
        dma_cfg(DMA_L2_1|DMA_FP32,DMA_GM|DMA_FP16|highptr(p),0.f);dma_cfg_2d(count,128,n*2);dma_i_2d((float*)(16384+buf*4096+h*2048),lowptr(p),64);
    }
#endif
}
