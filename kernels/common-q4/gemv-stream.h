#include "common-q4/source/adapt-gemv/bench/kernels/cdnn_ref.h"
#include "common-q4/source/adapt-gemv/experiments/weight-ds/activation-stage.h"
#include "common-q4/source/adapt-gemv/experiments/weight-ds/transfers/weight_dma-dual16-normal-38ba6b5b.h"
#include "common-q4/source/adapt-gemv/experiments/weight-ds/transfers/weight_ds-dual16-normal-65ea075b.h"
#include "common-q4/source/adapt-gemv/experiments/weight-ds/transfers/weight_dma_second-dual16-normal-a8574bc2.h"
#include "common-q4/source/adapt-gemv/experiments/weight-ds/transfers/weight_ds_second-dual16-normal-cca2785d.h"
#include "common-q4/source/adapt-gemv/experiments/weight-ds/transfers/mac_group-batch16-normal-584ec4a6.h"
#include "common-q4/source/adapt-gemv/experiments/weight-ds/transfers/scale_dma-dual16-normal-0119e532.h"
#include "common-q4/source/adapt-gemv/experiments/weight-ds/transfers/scale_ds-dual16-normal-d68effc0.h"
#include "common-q4/source/adapt-gemv/experiments/weight-ds/transfers/ew_group-base-normal-9c3f8f1a.h"
#include "common-q4/source/adapt-gemv/experiments/weight-ds/transfers/rs_group-base-normal-9099e3a9.h"
#include "common-q4/source/adapt-gemv/experiments/weight-ds/transfers/output_dma-base-normal-124c2a3c.h"
#ifndef STREAM_PROFILE
#define STREAM_PROFILE 0
#endif
#if STREAM_PROFILE
#define STREAM_MEASURE(slot,body) {unsigned int stamp=(unsigned int)get_clock();body;timing[slot]+=(unsigned int)get_clock()-stamp;}
#define STREAM_COUNT(slot) {++timing[slot];}
#else
#define STREAM_MEASURE(slot,body) {body;}
#define STREAM_COUNT(slot) {}
#endif
template<int cid> static __device__ __attribute__((noinline)) void stream_owner(
 _global_ptr_ float* partial,_global_ptr_ const int8_t* weights,_global_ptr_ const unsigned short* scales,
 int n,int cl,int nc,int blocks,int tiles,int cache_base,int resident,float a0,float a1,unsigned int* timing){
    const int batch=1,k=blocks*32;
    int engine=cid==0?DMAIN_0:cid==1?DMAIN_1:cid==2?MAC:cid==3?DS_0:cid==4?DS_1:cid==5?EW:cid==6?RS:DMAOUT;
    int delay=cid==3||cid==4?1:cid==2?2:cid==5?3:cid==6?4:cid==7?5:0;
        int tasks=((resident+15)/16)*tiles,held=0,item=0,base=cache_base;
        for(int step=0;step<tasks+5;++step){
            if(held){STREAM_MEASURE(2,xfence_unlock(engine));held=0;}
            STREAM_MEASURE(3,sync());
            int t=step-delay;if(t<0 || t>=tasks)continue;
            int count=blocks-base;if(count>16)count=16;
            int buf=t&1,tile=4*(cl+(item/2)*nc)+2*(item%2);
            STREAM_MEASURE(1,{
                xfence_lock(engine);
                if(cid==0)weight_dma(gptr_i8(weights,((long long)tile*blocks+base)*512),buf,count,blocks);
                if(cid==1){weight_dma_second(gptr_i8(weights,((long long)tile*blocks+base)*512),buf,count,blocks);commonv_s_dma(scales,tile*32,base,k,n,buf,count,t);}
                if(cid==3)weight_ds(buf,count);
                if(cid==4)commonv_scale_ds(buf,count,batch,t%3,t);
                if(cid==2)mac_group(buf,count,batch,a0,a1,base-cache_base);
                if(cid==5)ew_group(buf,count,batch,0,t%3);
                if(cid==6)rs_group(buf,count,batch);
                if(cid==7)output_dma(gptr_f32(partial,((long long)tile*blocks+base)*32),buf,count,batch,blocks);
            });
            STREAM_COUNT(5);held=1;
            if(++item==tiles){item=0;base+=16;}
        }
        if(held){STREAM_MEASURE(2,xfence_unlock(engine));}
}
__global__ void Q4_STREAM_ENTRY(float *partial,const int8_t *weights,const int8_t *activation,
 int k,int n,int batch,int first,int rows,int components,int bits,float a0,float a1,float unused,
 int pipelined,const unsigned short *scales,int fold,const float *device_scales) {
    if(batch!=1 || fold || !pipelined || core_num()!=8 || components!=2 || bits!=8 ||
       n%256 || rows%256 || k<512 || k>16384 || k%256 || first!=0 || Q4_ACT_CACHE_BLOCKS>128)return;
    __local__ __attribute__((aligned(64))) float act_parameters[16];
    int cid=core_id(),cl=cluster_id(),nc=cluster_num(),blocks=k/32;
    if(cid==2){GM2LM(device_scales,act_parameters,64);a0=act_parameters[0];a1=act_parameters[1];}
#if STREAM_PROFILE
    __local__ __attribute__((aligned(64))) unsigned int timing[16];for(int j=0;j<16;++j)timing[j]=0;
    unsigned int total_start=(unsigned int)get_clock();
#endif
    unsigned int* dt=nullptr;
#if STREAM_PROFILE
    dt=timing;
#endif
    int tiles=2*((rows/128+nc-1-cl)/nc);
    for(int cache_base=0;cache_base<blocks;cache_base+=Q4_ACT_CACHE_BLOCKS){
        int resident=blocks-cache_base;if(resident>Q4_ACT_CACHE_BLOCKS)resident=Q4_ACT_CACHE_BLOCKS;
        // No engine consumes L1D until all resident windows have completed.
        STREAM_MEASURE(0,{
            if(cid==0)for(int local=0;local<resident;local+=16){
                int count=resident-local;if(count>16)count=16;
                next_activation_stage(activation,k,cache_base+local,count,local);
            }
        });
        STREAM_MEASURE(3,sync());STREAM_COUNT(6);
        if(cid==0)stream_owner<0>(partial,weights,scales,n,cl,nc,blocks,tiles,cache_base,resident,a0,a1,dt);
        else if(cid==1)stream_owner<1>(partial,weights,scales,n,cl,nc,blocks,tiles,cache_base,resident,a0,a1,dt);
        else if(cid==2)stream_owner<2>(partial,weights,scales,n,cl,nc,blocks,tiles,cache_base,resident,a0,a1,dt);
        else if(cid==3)stream_owner<3>(partial,weights,scales,n,cl,nc,blocks,tiles,cache_base,resident,a0,a1,dt);
        else if(cid==4)stream_owner<4>(partial,weights,scales,n,cl,nc,blocks,tiles,cache_base,resident,a0,a1,dt);
        else if(cid==5)stream_owner<5>(partial,weights,scales,n,cl,nc,blocks,tiles,cache_base,resident,a0,a1,dt);
        else if(cid==6)stream_owner<6>(partial,weights,scales,n,cl,nc,blocks,tiles,cache_base,resident,a0,a1,dt);
        else if(cid==7)stream_owner<7>(partial,weights,scales,n,cl,nc,blocks,tiles,cache_base,resident,a0,a1,dt);
        // Drain the existing engine schedule before replacing resident L1D.
        STREAM_MEASURE(3,sync());
    }
#if STREAM_PROFILE
    timing[4]=(unsigned int)get_clock()-total_start;
    LM2GM(timing,(_global_ptr_ unsigned int*)gptr_f32(partial,(long long)rows*blocks)+(cl*8+cid)*16,64);
#endif
}
