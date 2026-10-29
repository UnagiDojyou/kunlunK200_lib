#if K_PARTS > 1
#include "source/adapt-gemm/combo-ops.h"
#else
#include "source/adapt-gemm/big-ops.h"
#endif
#include "common-feed.h"
// The L1E layout is identical to the existing compact-scale M128 path.
enum { owners_blocks=4, owners_plane=256, owners_scale_slot=8,
       owners_acc=2*owners_blocks*owners_plane+3*owners_scale_slot,
       owners_row_scale=owners_acc+owners_blocks*owners_plane };
template<int cid> static __device__ __attribute__((noinline)) void m128_owner(
 _global_ptr_ const int8_t* w,_global_ptr_ const unsigned short* scales,int n0,int kp,int np,int start_g,int nk,unsigned int* dt){
 const int engine=cid==0?DMAIN_0:cid==1?DMAIN_1:cid==2?MAC:cid==3?DS_0:cid==4?DS_1:EW;
 const int delay=cid==3||cid==4?1:cid==2?2:cid==5?3:0;
 int held=0;
 for(int step=0;step<nk+3;++step){
  if(held){xfence_unlock(engine);held=0;}sync();
  int g=step-delay;if(g<0||g>=nk)continue;
  int buf=g&1;xfence_lock(engine);
  if(cid==0)common_w_dma(w,n0,(start_g+g)*128,kp,buf,128,32);
  if(cid==1)common_s_dma(scales,n0,(start_g+g)*4,kp,np,buf,4,32);
  if(cid==3)common_w_ds(buf,128,32);
  if(cid==4)common_s_ds(buf,4,128,32,g%3);
  if(cid==2)q40_mac(g*512,buf,128,32,128,buf);
  if(cid==5)q40_reduce(buf,g%3,4,128,32,owners_acc,g==0,dt);
  held=1;
 }
 if(held){xfence_unlock(engine);}
}
static __device__ __attribute__((noinline)) void m128_idle(int nk){for(int step=0;step<nk+3;++step)sync();}
__global__ void OWNER_ENTRY(float* out,const int8_t* a,const int8_t* w,const unsigned short* scales,const float* row_scale,
 int m,int np,int kp,int nt,int kt,unsigned int* stats){
 if(cluster_num()!=4||core_num()!=8||np<=0||m!=128||nt!=32||kt!=128||np%32||kp%128||kp>16384||kp<=0)return;
 const int local_nc=cluster_num()/K_PARTS,kr=cluster_id()/local_nc,rank=cluster_id()%local_nc;
 const int start_g=(kp/128)*kr/K_PARTS,end_g=(kp/128)*(kr+1)/K_PARTS,nk=end_g-start_g,kpart=nk*128;
 if(kpart<=0||128*kpart>524288)return;
 const int cid=core_id(),count=(np/32+local_nc-1-rank)/local_nc;
 if(cid==0)for(int kb=0;kb<kpart/32;++kb){
  RUN_ENGINE(0,DMAIN_0,big_load_a(a,0,start_g*128+kb*32,kp,4));
  RUN_ENGINE(2,DS_0,big_shuffle_a(kb*128,4));
 }
 JOIN();
#if K_PARTS == 1
 if(cid==1){RUN_ENGINE(11,DMAIN_1,q40_row_scale_dma(row_scale,128));RUN_ENGINE(12,DS_1,q40_row_scale_ds(128,32,owners_row_scale));}
#endif
 JOIN();
 for(int task=0;task<count;++task){
  int n0=(rank+task*local_nc)*32;unsigned int* dt=nullptr;
  if(cid==0)m128_owner<0>(w,scales,n0,kp,np,start_g,nk,dt);
  else if(cid==1)m128_owner<1>(w,scales,n0,kp,np,start_g,nk,dt);
  else if(cid==2)m128_owner<2>(w,scales,n0,kp,np,start_g,nk,dt);
  else if(cid==3)m128_owner<3>(w,scales,n0,kp,np,start_g,nk,dt);
  else if(cid==4)m128_owner<4>(w,scales,n0,kp,np,start_g,nk,dt);
  else if(cid==5)m128_owner<5>(w,scales,n0,kp,np,start_g,nk,dt);
  else m128_idle(nk);
  JOIN();
  if(cid==5){RUN_ENGINE(5,EW,q40_finish(128,32,owners_acc,owners_row_scale,4));}JOIN();
  if(cid==6){RUN_ENGINE(6,RS,q40_rs(128,128,32));}JOIN();
  if(cid==7){RUN_ENGINE(7,DMAOUT,stage_out(gptr_f32(out,(long long)kr*128*np+n0),128,32,np));}JOIN();
 }
}
