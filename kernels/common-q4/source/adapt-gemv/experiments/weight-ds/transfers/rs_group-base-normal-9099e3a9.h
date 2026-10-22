static inline __device__ void rs_group(int buf,int count,int batch) {
    v16f32 *e=&((v16f32 *)0)[buf*128];v16f32 *r=(v16f32 *)(buf*8192);
    rs_row_cfg_stride(1,4);rs_row_cfg_loop(count);
    for(int i=0;i<4;++i)rs_row_batch(&r[i][0],&e[i*count][0],NBANKS);
}
