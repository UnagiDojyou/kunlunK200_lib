static inline __device__ void scale_ds(int buf,int count,int batch,int slot) {
    weight_ds_second(buf,count);
    v16f32 *s=&((v16f32 *)0)[256+slot*64];
    ds_shuffle_cfg_datatype(DS_FP32);ds_shuffle_cfg_blkx(32);ds_cfg_output_bank(NBANKS);
    for(int h=0;h<2;++h)for(int i=0;i<2;++i){
        float *p=(float *)(16384+buf*4096+h*2048);
        ds_shuffle_coa(count,&s[(h*2+i)*count][0],p+i*16,DS_L1E);
    }
}

static inline __device__ void commonv_scale_ds(int buf,int count,int batch,int slot,int item){
#if GEMV_CACHE128
    weight_ds_second(buf,count);v16f32* s=&((v16f32*)0)[256+slot*64];
    ds_shuffle_cfg_datatype(DS_FP32);ds_shuffle_cfg_blkx(128);ds_cfg_output_bank(NBANKS);
    for(int h=0;h<2;++h)for(int i=0;i<2;++i)ds_shuffle_coa(count,&s[(h*2+i)*count][0],(float*)(16384+((item/2)&1)*8192)+((item%2)*64+h*32+i*16),DS_L1E);
#else
    scale_ds(buf,count,batch,slot);
#endif
}
