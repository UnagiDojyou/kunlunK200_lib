#include "combo-ops.h"
#include "../../common-feed.h"
#ifndef Q40_SERIAL
#define Q40_SERIAL 0
#endif
__global__ void q40_gemm(float* out,const int8_t* a,const int8_t* w,const unsigned short* scales,const float* row_scale,
                         int m,int np,int kp,int nt,int kt,unsigned int* stats){
    int local_nc=cluster_num()/K_PARTS,kr=cluster_id()/local_nc,local_cl=cluster_id()%local_nc;
    int start_g=(kp/kt)*kr/K_PARTS,end_g=(kp/kt)*(kr+1)/K_PARTS,nk=end_g-start_g,kpart=nk*kt;
    int mt=TILE_M;while(mt>32&&mt*kpart>524288)mt/=2;
    int phys=m<=32?m:mt,blocks=kt/32,nm=(m+mt-1)/mt;
    int rg=(phys+31)/32;
    int sz=phys*(nt/16),scale_slot=SCALE_COMPACT?blocks*(nt/16):blocks*sz;
    if(core_num()!=8||m<1||m>128||nt<32||nt>128||nt%32||(kt!=64&&kt!=128&&kt!=256)||np%nt||kp%kt||kp>16384||
       rg*32*kpart>524288||phys*nt*4>16384||kt*nt>32768||(!SCALE_COMPACT&&2*blocks*phys*nt*4>32768)||(2*blocks*sz+3*scale_slot+(!EW_FUSED?blocks*sz:0)+(nm*rg*32*kpart<=524288?nm:1)*(blocks+1)*sz)*64>262144)return;
    int cid=core_id(),cl=cluster_id(),nc=local_nc,nn=np/nt;
    int reuse=nm*rg*32*kpart<=524288,lanes=reuse?nc:(nm<nc&&nc%nm==0?nc/nm:nm>=nc?1:nc),rank=local_cl%lanes;
    int work_m=reuse?nm:1,plane=phys*(nt/16),acc_base=2*blocks*plane+3*scale_slot+(!EW_FUSED?blocks*plane:0),row_base=acc_base+work_m*blocks*plane;
#if GEMM_PROFILE
    __local__ __attribute__((aligned(64))) unsigned int ticks[32];for(int i=0;i<32;++i)ticks[i]=0;unsigned int begin=(unsigned int)get_clock();
#endif
    unsigned int* dt=nullptr;
#if GEMM_PROFILE
    dt=ticks;
#endif
    for(int outer=0;outer<(reuse?1:nm);++outer){
        if(!reuse&&(nm>=nc?outer%nc!=local_cl:(nc%nm==0&&outer!=local_cl/lanes)))continue;
        if(cid==0)for(int mi=0;mi<work_m;++mi)for(int kb=0;kb<kpart/32;++kb){
            RUN_ENGINE(0,DMAIN_0,big_load_a(a,(reuse?mi:outer)*mt,(start_g*kt+kb*32),kp,rg));
            RUN_ENGINE(2,DS_0,big_shuffle_a((mi*(kpart/32)+kb)*rg*32,rg));
        }
        JOIN();
        JOIN();
        int count=(nn+lanes-1-rank)/lanes;
        for(int task=0;task<count;++task){
            int n0=(rank+task*lanes)*nt;
            _global_ptr_ const int8_t* wp=gptr_i8(w,(long long)(n0/nt)*(kp/2)*nt);
            _global_ptr_ const unsigned short* sp=hptr(scales,(long long)(n0/nt)*(kp/32)*nt);
            if(Q40_SERIAL){
                for(int g=0;g<nk;++g){
                    if(cid==0){
                        RUN_ENGINE(1,DMAIN_0,common_w_dma(w,n0,((start_g+g))*kt,kp,0,kt,nt));
                        RUN_ENGINE(11,DMAIN_1,common_s_dma(scales,n0,((start_g+g))*blocks,kp,np,0,blocks,nt));
                        RUN_ENGINE(3,DS_0,common_w_ds(0,kt,nt));RUN_ENGINE(12,DS_1,common_s_ds(0,blocks,phys,nt,0));
                        for(int mi=0;mi<work_m;++mi){
                            RUN_ENGINE(4,MAC,q40_mac((mi*kpart+g*kt)*rg,0,kt,nt,phys,0));
                            RUN_ENGINE(5,EW,q40_reduce(0,0,blocks,phys,nt,acc_base+mi*blocks*plane,g==0,dt));
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
                        if(cid==0)common_w_dma(w,n0,((start_g+g))*kt,kp,buf,kt,nt);
                        if(cid==1)common_s_dma(scales,n0,((start_g+g))*blocks,kp,np,buf,blocks,nt);
                        if(cid==3)common_w_ds(buf,kt,nt);
                        if(cid==4)common_s_ds(buf,blocks,phys,nt,g%3);
                        if(cid==2)q40_mac((mi*kpart+g*kt)*rg,buf,kt,nt,phys,item&1);
                        if(cid==5)q40_reduce(item&1,g%3,blocks,phys,nt,acc_base+mi*blocks*plane,g==0,dt);
                    ISSUE_END(cid)
                    });held=1;
                }
                if(held){MEASURE(slot,{DETAIL(wait_slot(cid),xfence_unlock(engine));});}JOIN();
            }
            for(int mi=0;mi<work_m;++mi){
                int m0=(reuse?mi:outer)*mt,valid=m-m0;if(valid>phys)valid=phys;
                if(cid==5){RUN_ENGINE(5,EW,q40_finish(phys,nt,acc_base+mi*blocks*plane,row_base+mi*plane,blocks));}JOIN();
                if(cid==6){RUN_ENGINE(6,RS,q40_rs(phys,valid,nt));}JOIN();
                if(cid==7){RUN_ENGINE(7,DMAOUT,stage_out(gptr_f32(out,((long long)kr*m+m0)*np+n0),valid,nt,np));}JOIN();
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
