static inline __device__ void output_dma(_global_ptr_ float *out,int buf,int count,int batch,int blocks) {
    // Preserve the original [N32 tile,K32 block,32] partial layout. Source
    // strides are 256B; destination strides 128B. Each starting address is 64B aligned.
    for(int h=0;h<2;++h){
        _global_ptr_ float *p=gptr_f32(out,(long long)h*blocks*32);
        dma_cfg(DMA_GM|DMA_FP32|highptr(p),DMA_L2R|DMA_FP32,0.f);
        dma_cfg_2d(count,128,256);
        dma_o_2d(lowptr(p),(v16f32 *)(buf*8192+h*128),128);
    }
}
