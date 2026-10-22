#include "m8-ops.h"
#include "../../common-feed.h"
#ifndef Q40_SERIAL
#define Q40_SERIAL 0
#endif
__global__ void q40_gemm(float* out,const int8_t* a,const int8_t* w,const unsigned short* scales,const float* row_scale,
                         int m,int np,int kp,int nt,int kt,unsigned int* stats){
    int phys=m<32?m:32,blocks=kt/32,nm=(m+31)/32;
    int sz=phys*(nt/16),acc_lanes=FOLD_LANES?blocks/2:blocks;
    int scale_faces=(nm>=2&&nm*32*kp<=524288)?2:3,scale_slot=blocks*sz;
    if(core_num()!=8||m<1||m>8||nt<32||nt>512||nt%32||(kt!=64&&kt!=128&&kt!=256&&kt!=512)||np%nt||kp%kt||kp>16384||
       32*kp>524288||phys*nt*4>16384||kt*nt>32768||2*blocks*nt*4>32768||(2*blocks*sz+scale_faces*scale_slot+(nm*32*kp<=524288?nm:1)*(acc_lanes+1)*sz)*64>262144)return;
    int cid=core_id(),cl=cluster_id(),nc=cluster_num(),nn=np/nt,nk=kp/kt;
    int reuse=nm*32*kp<=524288,lanes=reuse?nc:(nm<nc&&nc%nm==0?nc/nm:nm>=nc?1:nc),rank=cl%lanes;
    int work_m=reuse?nm:1,plane=phys*(nt/16),acc_base=2*blocks*plane+scale_faces*scale_slot,row_base=acc_base+work_m*acc_lanes*plane;
#if GEMM_PROFILE
    __local__ __attribute__((aligned(64))) unsigned int ticks[32];for(int i=0;i<32;++i)ticks[i]=0;unsigned int begin=(unsigned int)get_clock();
#endif
    unsigned int* dt=nullptr;
#if GEMM_PROFILE
    dt=ticks;
#endif
    for(int outer=0;outer<(reuse?1:nm);++outer){
        if(!reuse&&(nm>=nc?outer%nc!=cl:(nc%nm==0&&outer!=cl/lanes)))continue;
        if(A_BLOCK){
            if(cid==0)for(int off=0;off<kp;off+=A_BLOCK){
                int width=kp-off;if(width>A_BLOCK)width=A_BLOCK;
                RUN_ENGINE(0,DMAIN_0,preload_a(a,off,kp,width,off==0||width!=A_BLOCK));
                RUN_ENGINE(2,DS_0,shuffle_a(off,width));
            }
        }else{
        if(cid==0)for(int mi=0;mi<work_m;++mi)for(int g=0;g<nk;++g){
            RUN_ENGINE(0,DMAIN_0,load_a(a,(reuse?mi:outer)*32,g*kt,kp,kt));RUN_ENGINE(2,DS_0,shuffle_a(mi*kp+g*kt,kt));
        }
        }
        JOIN();
        if(cid==1)for(int mi=0;mi<work_m;++mi){
            RUN_ENGINE(11,DMAIN_1,q40_row_scale_dma(gptr_f32c(row_scale,(long long)(reuse?mi:outer)*32*16),phys));
            RUN_ENGINE(12,DS_1,q40_row_scale_ds(phys,nt,row_base+mi*plane));
        }
        JOIN();
        int count=(nn+lanes-1-rank)/lanes;
        for(int task=0;task<count;++task){
            int n0=(rank+task*lanes)*nt;
            _global_ptr_ const int8_t* wp=gptr_i8(w,(long long)(n0/nt)*(kp/2)*nt);
            _global_ptr_ const unsigned short* sp=hptr(scales,(long long)(n0/nt)*(kp/32)*nt);
            if(Q40_SERIAL){
                for(int g=0;g<nk;++g){
                    if(cid==0){
                        RUN_ENGINE(1,DMAIN_0,common_w_dma(w,n0,(g)*kt,kp,0,kt,nt));
                        RUN_ENGINE(11,DMAIN_1,common_s_dma(scales,n0,(g)*blocks,kp,np,0,blocks,nt));
                        RUN_ENGINE(3,DS_0,common_w_ds(0,kt,nt));RUN_ENGINE(12,DS_1,common_s_ds(0,blocks,phys,nt,0));
                        for(int mi=0;mi<work_m;++mi){
                            RUN_ENGINE(4,MAC,q40_mac(mi*kp+g*kt,0,kt,nt,phys,0));
                            RUN_ENGINE(5,EW,q40_reduce(0,0,blocks,phys,nt,acc_base+mi*acc_lanes*plane,g==0,dt));
                        }
                    }
                    JOIN();
                }
            }else{
                int engine=-1,delay=0,slot=0;
                if(cid==0){engine=DMAIN_0;slot=1;}if(cid==1){engine=DMAIN_1;slot=11;}
                if(cid==3){engine=DS_0;delay=1;slot=3;}if(cid==4){engine=DS_1;delay=1;slot=12;}
                if(cid==2){engine=MAC;delay=2;slot=4;}if(cid==5){engine=EW;delay=3;slot=5;}
                int held=0,items=nk*work_m;
                for(int step=0;step<items+3;++step){
                    if(held){MEASURE(slot,{DETAIL(wait_slot(cid),xfence_unlock(engine));});held=0;}JOIN();COUNT(24);int item=step-delay;
                    if(engine<0||item<0||item>=items)continue;
                    int g=item/work_m,mi=item%work_m,buf=g&1;
                    if(cid!=2&&cid!=5&&mi!=0)continue;
                    MEASURE(slot,{
                        DETAIL(cid==5?16:cid==2?19:31,xfence_lock(engine));
                        ISSUE_BEGIN
                        if(cid==0)common_w_dma(w,n0,(g)*kt,kp,buf,kt,nt);
                        if(cid==1)common_s_dma(scales,n0,(g)*blocks,kp,np,buf,blocks,nt);
                        if(cid==3)common_w_ds(buf,kt,nt);
                        if(cid==4)common_s_ds(buf,blocks,phys,nt,g%scale_faces);
                        if(cid==2)q40_mac(mi*kp+g*kt,buf,kt,nt,phys,item&1);
                        if(cid==5)q40_reduce(item&1,g%scale_faces,blocks,phys,nt,acc_base+mi*acc_lanes*plane,g==0,dt);
                    ISSUE_END(cid)
                    });held=1;
                }
                if(held){MEASURE(slot,{DETAIL(wait_slot(cid),xfence_unlock(engine));});}JOIN();
            }
            for(int mi=0;mi<work_m;++mi){
                int m0=(reuse?mi:outer)*32,valid=m-m0;if(valid>32)valid=32;
                if(cid==5){RUN_ENGINE(5,EW,q40_finish(phys,nt,acc_base+mi*acc_lanes*plane,row_base+mi*plane,acc_lanes));}JOIN();
                if(cid==6){RUN_ENGINE(6,RS,q40_rs(phys,valid,nt));}JOIN();
                if(cid==7){RUN_ENGINE(7,DMAOUT,stage_out(gptr_f32(out,(long long)m0*np+n0),valid,nt,np));}JOIN();
            }
#if GEMM_PROFILE
            ticks[10]++;
#endif
        }
    }
#if GEMM_PROFILE
    ticks[9]=(unsigned int)get_clock()-begin;LM2GM(ticks,stats+(cl*8+cid)*32,128);
#endif
}
