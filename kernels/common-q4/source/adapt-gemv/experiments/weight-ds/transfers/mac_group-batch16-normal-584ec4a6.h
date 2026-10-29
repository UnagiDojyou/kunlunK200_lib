static inline __device__ void mac_group(int buf,int count,int batch,float a0,float a1,int activation_base=0) {
    v16f32 *e=&((v16f32 *)0)[buf*128];
    mm_cfg_stride(1);mm_cfg_basic_int8(32,1,count);
    for(int b=0;b<count;++b){
        v32i8 *w=&((v32i8 *)0)[buf*512+b*16];
        mm_cfg_dequant_scale(a0);
        mm_int4(&e[b],&((v32i8 *)0)[(activation_base+b)*128],w,MAC_INT8_D8W4);
        mm_int4(&e[count*2+b],&((v32i8 *)0)[(activation_base+b)*128],&w[256],MAC_INT8_D8W4);
        mm_cfg_dequant_scale(a1);
        mm_acc_int4(&e[b],&((v32i8 *)0)[(activation_base+b)*128+64],w,MAC_INT8_D8W4);
        mm_acc_int4(&e[count*2+b],&((v32i8 *)0)[(activation_base+b)*128+64],&w[256],MAC_INT8_D8W4);
    }
}
