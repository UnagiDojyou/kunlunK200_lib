#pragma once
#ifndef NEXT_MIN_CLEAR
#define NEXT_MIN_CLEAR 0
#endif
#ifndef NEXT_A_2D
#define NEXT_A_2D 0
#endif
// Only M=1; initialize every L2 byte even for a short K-window. The valid
// row is independent of neighbouring inactive M lanes populated by DS.
static inline __device__ void next_activation_stage(_global_ptr_ const int8_t *activation,int k,int base,int count,int destination=0){
    _global_ptr_ const int8_t *zero=gptr_i8(activation,(long long)4*k);
    for(int j=(NEXT_MIN_CLEAR && count==16?2:0);j<3;++j)dma_in_1d((int8_t *)(j*1024),zero,1024,DMA_L2_0,DMA_INT8);
#if NEXT_A_2D
    _global_ptr_ const int8_t *a=gptr_i8(activation,(long long)base*64);
    xfence_lock(DMAIN_0);dma_cfg(DMA_L2_0|DMA_INT8,DMA_GM|DMA_INT8|highptr(a),0.f);
    dma_cfg_2d(2,1024,2*k);dma_i_2d((int8_t *)0,lowptr(a),count*64);xfence_unlock(DMAIN_0);
#else
    for(int c=0;c<2;++c){
        _global_ptr_ const int8_t *a=gptr_i8(activation,(long long)c*2*k+base*64);
        dma_in_1d((int8_t *)(c*1024),a,count*64,DMA_L2_0,DMA_INT8);
    }
#endif
    for(int start=0;start<count*2;start+=NEXT_DS_PAIRS){
        xfence_lock(DS_0);
        ds_shuffle_cfg_datatype(DS_INT8);ds_shuffle_cfg_blkx(32);ds_cfg_output_bank(NBANKS);
        for(int j=start;j<count*2 && j<start+NEXT_DS_PAIRS;++j){
            int b=j/2,c=j&1;v32i8 *a=&((v32i8 *)0)[(destination+b)*128+c*64];int8_t *p=(int8_t *)(c*1024+b*64);
            ds_shuffle_batch(&a[0][0],p,32,DS_L1D);
            ds_shuffle_batch(&a[0][1],p+512,32,DS_L1D);
        }
        xfence_unlock(DS_0);
    }
}
