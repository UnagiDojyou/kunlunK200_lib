static inline __device__ void weight_ds(int buf,int count){
    ds_shuffle_cfg_datatype(DS_INT16);ds_cfg_output_bank(NBANKS);
#if COMMON_LAYOUT==2
    ds_shuffle_cfg_blkx(count*16);ds_shuffle_batch(&((v32i8*)0)[buf*512][0],(int8_t*)(buf*16384),count*16,DS_L1W);
#elif COMMON_LAYOUT==1
    ds_shuffle_cfg_blkx(16);for(int b=0;b<count;++b)ds_shuffle_batch(&((v32i8*)0)[buf*512+b*16][0],(int8_t*)(buf*16384+b*512),16,DS_L1W);
#else
    ds_shuffle_cfg_blkx(16);ds_shuffle_coa(count*16,&((v32i8*)0)[buf*512][0],(int8_t*)(buf*16384),DS_L1W);
#endif
}
