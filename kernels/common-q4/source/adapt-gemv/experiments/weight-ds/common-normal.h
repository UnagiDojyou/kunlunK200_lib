// Engine-owner schedule adapted from sam6321/KunlunK200,
// commit 5ead5bb10ab6d747e0f33b739826f111e42ec06a (MIT; see reference/LICENSE).
// Unlike that GEMM, weights stream for every N tile. Original Q4 scales remain
// exact FP16 metadata, applied by SD EW before the CL partial reduction.
#include "../../bench/kernels/cdnn_ref.h"
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

#include "transfers/weight_dma-dual16-normal-38ba6b5b.h"
#include "transfers/weight_ds-dual16-normal-65ea075b.h"
#include "transfers/weight_dma_second-dual16-normal-a8574bc2.h"
#include "transfers/weight_ds_second-dual16-normal-cca2785d.h"
#include "transfers/mac_group-batch16-normal-584ec4a6.h"
// Scale L1E lifetime spans DS->MAC->EW: THREE slots are required.
// FP16 weights remain 4.5bpw: DMA converts metadata into L2_1 FP32 on chip.
#include "transfers/scale_dma-dual16-normal-0119e532.h"
#include "transfers/scale_ds-dual16-normal-d68effc0.h"
#include "transfers/ew_group-base-normal-9c3f8f1a.h"
#include "transfers/rs_group-base-normal-9099e3a9.h"
#include "transfers/output_dma-base-normal-124c2a3c.h"
__global__ void Q4_PIPE_ENTRY(float *partial,const int8_t *weights,const int8_t *activation,
                         int k,int n,int batch,int first,int rows,int components,int bits,
                         float a0,float a1,float unused,int pipelined,const unsigned short *scales,int fold
#if NEXT_DEVICE_SCALE
                         ,const float *device_scales
#endif
                         ) {
    if(Q4_PIPE_GUARD || core_num()!=8 || components!=2 || bits!=8 || n%256 || rows%256)return;
#if NEXT_DEVICE_SCALE
    __local__ __attribute__((aligned(64))) float act_parameters[16];
    if(core_id()==2){GM2LM(device_scales,act_parameters,64);a0=act_parameters[0];a1=act_parameters[1];}
#endif
    int cid=core_id(),cl=cluster_id(),nc=cluster_num(),blocks=k/32;
    int window=Q4_PIPE_WINDOW;
    int tiles=GEMV_CACHE128?2*((rows/128+nc-1-cl)/nc):(rows/64)/nc,groups=(blocks+7)/8;
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
                    int tile=GEMV_CACHE128?4*(cl+(t/2)*nc)+2*(t%2):2*(cl+t*nc);
                    _global_ptr_ const int8_t *w=gptr_i8(weights,((long long)(first/32+tile)*blocks+base)*512);
                    _global_ptr_ float *p=gptr_f32(partial,((long long)tile*(fold?groups:blocks)+(fold?base/8:base))*batch*32);
                    xfence_lock(DMAIN_0);weight_dma(w,0,count,blocks);xfence_unlock(DMAIN_0);
                    xfence_lock(DMAIN_1);commonv_s_dma(scales,first+tile*32,base,k,n,0,count,t);xfence_unlock(DMAIN_1);
                    xfence_lock(DS_0);weight_ds(0,count);xfence_unlock(DS_0);
                    xfence_lock(DS_1);commonv_scale_ds(0,count,batch,0,t);xfence_unlock(DS_1);
                    xfence_lock(MAC);mac_group(0,count,batch,a0,a1);xfence_unlock(MAC);
                    xfence_lock(EW);ew_group(0,count,batch,fold,0);xfence_unlock(EW);
                    xfence_lock(RS);rs_group(0,fold?1:count,batch);xfence_unlock(RS);
                    xfence_lock(DMAOUT);output_dma(p,0,fold?1:count,batch,blocks);xfence_unlock(DMAOUT);
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
            int buf=t&1,tile=GEMV_CACHE128?4*(cl+(t/2)*nc)+2*(t%2):2*(cl+t*nc);
            xfence_lock(engine);
            if(cid==0){weight_dma(gptr_i8(weights,((long long)(first/32+tile)*blocks+base)*512),buf,count,blocks);}
            if(cid==1){weight_dma_second(gptr_i8(weights,((long long)(first/32+tile)*blocks+base)*512),buf,count,blocks);commonv_s_dma(scales,first+tile*32,base,k,n,buf,count,t);}
            if(cid==3){weight_ds(buf,count);}
            if(cid==4){commonv_scale_ds(buf,count,batch,t%3,t);}
            if(cid==2){mac_group(buf,count,batch,a0,a1);}
            if(cid==5){ew_group(buf,count,batch,fold,t%3);}
            if(cid==6){rs_group(buf,fold?1:count,batch);}
            if(cid==7){output_dma(gptr_f32(partial,((long long)tile*(fold?groups:blocks)+(fold?base/8:base))*batch*32),buf,fold?1:count,batch,blocks);}
            held=1;
        }
        if(held){xfence_unlock(engine);}
        sync();
    }
}
