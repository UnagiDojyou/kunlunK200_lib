static inline __device__ void ew_group(int buf,int count,int batch,int fold,int slot) {
    v16f32 *e=&((v16f32 *)0)[buf*128],*s=&((v16f32 *)0)[256+slot*64];
    ew_cfg_activation_type(EW_NOACT);ew_cfg_coeff_scalar(0.f,0.f,1.f);
    ew_cfg_findmax(0,0.f);ew_cfg_stream_size(count*4);ew_cfg_col_size(NBANKS);
    ew_dsmul(e,e,s,EW_L2E);
}
