static inline __device__ void weight_dma(_global_ptr_ const int8_t *w,int buf,int count,int blocks) {
    for(int h=0;h<1;++h){
        _global_ptr_ const int8_t *p=gptr_i8(w,(long long)h*blocks*512);
        dma_cfg(DMA_L2_0|DMA_INT8,DMA_GM|DMA_INT8|highptr(p),0.f);
        dma_i((int8_t *)(buf*16384+h*8192),lowptr(p),count*512);
    }
}
