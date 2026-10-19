// Engine-owner schedule adapted from sam6321/KunlunK200,
// commit 5ead5bb10ab6d747e0f33b739826f111e42ec06a (MIT; see LICENSE).
// Unlike that GEMM, weights stream for every N tile. Original Q4 scales remain
// exact FP16 metadata, applied by SD EW before the CL partial reduction.
#include "sd-helpers.h"
#include "activation-stage.h"
#ifndef NEXT_COMPACT
#define NEXT_COMPACT 0
#endif
#ifndef NEXT_DEVICE_SCALE
#define NEXT_DEVICE_SCALE 0
#endif
#ifndef Q4_GROUPED_DS
#define Q4_GROUPED_DS 0
#endif

static inline __device__ void weight_dma(_global_ptr_ const int8_t *w,int buf,int count) {
    dma_cfg(DMA_L2_0|DMA_INT8,DMA_GM|DMA_INT8|highptr(w),0.f);
    dma_i((int8_t *)(buf*(Q4_PANEL_WIDTH*512)),lowptr(w),count*512);
}
static inline __device__ void weight_ds(int buf,int count) {
    ds_shuffle_cfg_datatype(DS_INT8);ds_shuffle_cfg_blkx(32);ds_cfg_output_bank(NBANKS);
    if(Q4_GROUPED_DS){
        v32i8 *w=&((v32i8 *)0)[buf*Q4_PANEL_WIDTH*64];int8_t *p=(int8_t *)(buf*(Q4_PANEL_WIDTH*512));
        ds_shuffle_coa(count*16,&w[0][0],p,DS_L1W);
        ds_shuffle_coa(count*16,&w[0][1],p+NBANKS,DS_L1W);return;
    }
    for(int b=0;b<count;++b){
        v32i8 *w=&((v32i8 *)0)[(buf*Q4_PANEL_WIDTH+b)*64];int8_t *p=(int8_t *)(buf*(Q4_PANEL_WIDTH*512)+b*512);
        ds_shuffle_coa(16,&w[0][0],p,DS_L1W);
        ds_shuffle_coa(16,&w[0][1],p+NBANKS,DS_L1W);
    }
}
static inline __device__ void mac_group(int buf,int count,int batch,float a0,float a1) {
    v16f32 *e=&((v16f32 *)0)[buf*128];
#if !NEXT_HOIST
    mm_cfg_stride(1);mm_cfg_basic_int8(32,batch,count*batch);
#endif
#if NEXT_PASS
    mm_cfg_dequant_scale(a0);
    for(int b=0;b<count;++b){
        v32i8 *w=&((v32i8 *)0)[buf*Q4_PANEL_WIDTH*64+b*16];
        mm_int4(&e[b*batch],&((v32i8 *)0)[b*128],w,MAC_INT8_D8W4);
    }
    mm_cfg_dequant_scale(a1);
    for(int b=0;b<count;++b){
        v32i8 *w=&((v32i8 *)0)[buf*Q4_PANEL_WIDTH*64+b*16];
        mm_acc_int4(&e[b*batch],&((v32i8 *)0)[b*128+64],w,MAC_INT8_D8W4);
    }
#else
    for(int b=0;b<count;++b){
        v32i8 *w=&((v32i8 *)0)[buf*Q4_PANEL_WIDTH*64+b*16];
        mm_cfg_dequant_scale(a0);mm_int4(&e[b*batch],&((v32i8 *)0)[b*128],w,MAC_INT8_D8W4);
        mm_cfg_dequant_scale(a1);mm_acc_int4(&e[b*batch],&((v32i8 *)0)[b*128+64],w,MAC_INT8_D8W4);
    }
#endif
}
// Scale L1E lifetime spans DS->MAC->EW: THREE slots are required.
// FP16 weights remain 4.5bpw: DMA converts metadata into L2_1 FP32 on chip.
static inline __device__ void scale_dma(_global_ptr_ const unsigned short *s,int buf,int count) {
    dma_cfg(DMA_L2_1|DMA_FP32,DMA_GM|DMA_FP16|highptr(s),0.f);
    dma_i((float *)(buf*4096),lowptr(s),count*64);
}
static inline __device__ void scale_ds(int buf,int count,int batch,int slot) {
    v16f32 *s=&((v16f32 *)0)[256+slot*64];float *p=(float *)(buf*4096);
    ds_shuffle_cfg_datatype(DS_FP32);ds_shuffle_cfg_blkx(32);ds_cfg_output_bank(NBANKS);
    for(int i=0;i<32;i+=NBANKS){
        if(batch==1){ds_shuffle_coa(count,&s[(i/NBANKS)*count][0],p+i,DS_L1E);}
        else{for(int b=0;b<count;++b)for(int m=0;m<batch;++m){
            ds_shuffle_coa(1,&s[(i/NBANKS)*count*batch+b*batch+m][0],p+b*32+i,DS_L1E);
        }}
    }
}
static inline __device__ void ew_group(int buf,int count,int batch,int fold,int slot) {
    v16f32 *e=&((v16f32 *)0)[buf*128],*s=&((v16f32 *)0)[256+slot*64];
    v16f32 *temp=&e[32],*acc=&e[64];
    ew_cfg_activation_type(EW_NOACT);ew_cfg_coeff_scalar(0.f,0.f,1.f);
    ew_cfg_findmax(0,0.f);ew_cfg_stream_size(count*batch*2);ew_cfg_col_size(NBANKS);
    if(!fold){ew_dsmul(e,e,s,EW_L2E);return;}
    ew_dsmul(temp,e,s,EW_L1E);xfence();
    if(fold==2){
        int live=count;
        ew_cfg_coeff_scalar(1.f,1.f,0.f);
        while(live>1){
            int half=live/2,left=live-half;ew_cfg_stream_size(half);
            ew_dsmadd(&temp[0],&temp[0],&temp[left],EW_L1E);
            ew_dsmadd(&temp[count],&temp[count],&temp[count+left],EW_L1E);
            xfence();live=left;
        }
        ew_cfg_coeff_scalar(1.f,0.f,0.f);ew_cfg_stream_size(1);
        ew_dsmadd(&e[0],&temp[0],&temp[0],EW_L2E);
        ew_dsmadd(&e[1],&temp[count],&temp[count],EW_L2E);return;
    }
    ew_cfg_stream_size(1);ew_cfg_coeff_scalar(1.f,0.f,0.f);
    ew_dsmadd(&acc[0],&temp[0],&temp[0],EW_L1E);ew_dsmadd(&acc[1],&temp[count],&temp[count],EW_L1E);xfence();
    ew_cfg_coeff_scalar(1.f,1.f,0.f);
    for(int b=1;b<count;++b){
        ew_dsmadd(&acc[0],&acc[0],&temp[b],EW_L1E);
        ew_dsmadd(&acc[1],&acc[1],&temp[count+b],EW_L1E);xfence();
    }
    ew_cfg_stream_size(2);ew_cfg_coeff_scalar(1.f,0.f,0.f);ew_dsmadd(e,acc,acc,EW_L2E);
}
static inline __device__ void rs_group(int buf,int count,int batch) {
    v16f32 *e=&((v16f32 *)0)[buf*128];v16f32 *r=(v16f32 *)(buf*8192);
    rs_row_cfg_stride(1,2);rs_row_cfg_loop(count*batch);
    for(int i=0;i<2;++i)rs_row_batch(&r[i][0],&e[i*count*batch][0],NBANKS);
}
static inline __device__ void output_dma(_global_ptr_ float *out,int buf,int count,int batch) {
    dma_cfg(DMA_GM|DMA_FP32|highptr(out),DMA_L2R|DMA_FP32,0.f);
    dma_cfg_2d(count*batch,128,128);
    dma_o_2d(lowptr(out),(v16f32 *)(buf*8192),128);
}
__global__ void Q4_PIPE_ENTRY(float *partial,const int8_t *weights,const int8_t *activation,
                         int k,int n,int batch,int first,int rows,int components,int bits,
                         float a0,float a1,float unused,int pipelined,const unsigned short *scales,int fold
#if NEXT_DEVICE_SCALE
                         ,const float *device_scales
#endif
                         ) {
    if(Q4_PIPE_GUARD || core_num()!=8 || components!=2 || bits!=8 || n%128 || rows%128)return;
#if NEXT_DEVICE_SCALE
    __local__ __attribute__((aligned(64))) float act_parameters[16];
    if(core_id()==2){GM2LM(device_scales,act_parameters,64);a0=act_parameters[0];a1=act_parameters[1];}
#endif
    int cid=core_id(),cl=cluster_id(),nc=cluster_num(),blocks=k/32;
    int window=Q4_PIPE_WINDOW;
    int tiles=(rows/32)/nc,groups=(blocks+7)/8;
    for(int base=0;base<blocks;base+=window){
        int count=blocks-base;if(count>window)count=window;
        if(cid==0){
            next_activation_stage(activation,k,base,count);
        }
#if NEXT_HOIST
        if(cid==2){xfence_lock(MAC);mm_cfg_stride(1);mm_cfg_basic_int8(32,batch,count*batch);xfence_unlock(MAC);}
#endif
        sync();
        if(!pipelined){
            if(cid==0){
                for(int t=0;t<tiles;++t){
                    int tile=cl+t*nc;
                    _global_ptr_ const int8_t *w=gptr_i8(weights,((long long)(first/32+tile)*blocks+base)*512);
                    _global_ptr_ float *p=gptr_f32(partial,((long long)tile*(fold?groups:blocks)+(fold?base/8:base))*batch*32);
                    xfence_lock(DMAIN_0);weight_dma(w,0,count);xfence_unlock(DMAIN_0);
                    xfence_lock(DMAIN_1);scale_dma((_global_ptr_ const unsigned short *)gptr_i16((_global_ptr_ const int16_t *)scales,((long long)(first/32+tile)*blocks+base)*32),0,count);xfence_unlock(DMAIN_1);
                    xfence_lock(DS_0);weight_ds(0,count);xfence_unlock(DS_0);
                    xfence_lock(DS_1);scale_ds(0,count,batch,0);xfence_unlock(DS_1);
                    xfence_lock(MAC);mac_group(0,count,batch,a0,a1);xfence_unlock(MAC);
                    xfence_lock(EW);ew_group(0,count,batch,fold,0);xfence_unlock(EW);
                    xfence_lock(RS);rs_group(0,fold?1:count,batch);xfence_unlock(RS);
                    xfence_lock(DMAOUT);output_dma(p,0,fold?1:count,batch);xfence_unlock(DMAOUT);
                }
            }
            sync();continue;
        }
        int engine=-1,delay=0;
        if(cid==0){engine=DMAIN_0;delay=0;}
        if(cid==1){engine=DMAIN_1;delay=0;}
        if(cid==3){engine=DS_0;delay=1;}
        if(cid==4){engine=DS_1;delay=1;}
        if(cid==2){engine=MAC;delay=2;}
        if(cid==5){engine=EW;delay=3;}
        if(cid==6){engine=RS;delay=4;}
        if(cid==7){engine=DMAOUT;delay=5;}
        int held=0;
        for(int step=0;step<tiles+5;++step){
            if(held){xfence_unlock(engine);held=0;}
            sync();
            int t=step-delay;
            if(engine<0 || t<0 || t>=tiles)continue;
            int buf=t&1,tile=cl+t*nc;
            xfence_lock(engine);
            if(cid==0){weight_dma(gptr_i8(weights,((long long)(first/32+tile)*blocks+base)*512),buf,count);}
            if(cid==1){scale_dma((_global_ptr_ const unsigned short *)gptr_i16((_global_ptr_ const int16_t *)scales,((long long)(first/32+tile)*blocks+base)*32),buf,count);}
            if(cid==3){weight_ds(buf,count);}
            if(cid==4){scale_ds(buf,count,batch,t%3);}
            if(cid==2){mac_group(buf,count,batch,a0,a1);}
            if(cid==5){ew_group(buf,count,batch,fold,t%3);}
            if(cid==6){rs_group(buf,fold?1:count,batch);}
            if(cid==7){output_dma(gptr_f32(partial,((long long)tile*(fold?groups:blocks)+(fold?base/8:base))*batch*32),buf,fold?1:count,batch);}
            held=1;
        }
        if(held){xfence_unlock(engine);}
        sync();
    }
}
