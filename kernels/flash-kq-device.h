#include "xpu/kernel/sdcdnn_header.h"
#include "flash-config.h"
static inline __device__ void load(_global_ptr_ const unsigned char *p,int rows,int k,int stride,int port,float peak){
 int engine=port?DMAIN_1:DMAIN_0,bank=port?DMA_L2_1:DMA_L2_0;
 xfence_lock(engine);dma_cfg(bank|DMA_INT16,DMA_GM|DMA_FP16|highptr(p),peak>0?peak:1.0f);
 dma_cfg_2d(rows,k*2,stride);dma_i_2d((void*)0,lowptr(p),k*2);xfence_unlock(engine);
 xfence_lock(port?DS_1:DS_0);ds_shuffle_cfg_datatype(DS_INT16);ds_shuffle_cfg_blkx(k);ds_cfg_output_bank(16);
 for(int j=0;j<(rows+15)/16;++j)ds_shuffle_batch(&((v16i16*)0)[j*k][0],(short*)0+j*16*k,k,port?DS_L1W:DS_L1D);
 xfence_unlock(port?DS_1:DS_0);
}

__global__ void KQ_NAME(float *score,const unsigned char *k,const unsigned char *q,const float *peaks,const float *ranges,const KunlunFlashConfig *cfg,int s0,int count){
 if(core_id()!=0)return;
 __local__ KunlunFlashConfig p;GM2LM(cfg,&p,sizeof(p));
 __local__ float a[8],r[8];GM2LM(peaks,a,32);for(int j=1;j<4;++j)if(a[j]>a[0])a[0]=a[j];
 int nt=count/32,mt=(p.m+p.qt-1)/p.qt,groups=p.h/p.hk;
 bool grouped=p.m==1&&groups>1&&p.tg_group;
 // Keep all four SDNNs useful even when there are fewer than four KV heads.
 int parts=(cluster_num()+p.b*p.hk-1)/(p.b*p.hk);if(parts>nt)parts=nt;
 for(int task=cluster_id();task<p.b*p.hk*parts;task+=cluster_num()){
  int kp=task%parts,hk=task/parts%p.hk,b=task/parts/p.hk;
  int first=(kp*nt/parts)*32,nkeys=((kp+1)*nt/parts)*32-first;
  // Each input DMA is bounded by the 128 KiB L2_1 bank. L1W keeps all chunks.
  for(int offset=0;offset<nkeys;){
   int kk=nkeys-offset,limit=65536/p.d;if(kk>limit)kk=limit;
   _global_ptr_ const unsigned char *src=k+(s0+first+offset)*p.k[1]+hk*p.k[2]+(b/(p.b/p.bk))*p.k[3];
   xfence_lock(DMAIN_1);dma_cfg(DMA_L2_1|DMA_INT16,DMA_GM|DMA_FP16|highptr(src),a[0]>0?a[0]:1.0f);dma_cfg_2d(kk,p.d*2,p.k[1]);dma_i_2d((void*)0,lowptr(src),p.d*2);xfence_unlock(DMAIN_1);
   xfence_lock(DS_1);ds_shuffle_cfg_datatype(DS_INT16);ds_shuffle_cfg_blkx(p.d);ds_cfg_output_bank(16);
   for(int j=0;j<kk/16;++j)ds_shuffle_batch(&((v16i16*)0)[(offset/16+j)*p.d][0],(short*)0+j*16*p.d,p.d,DS_L1W);xfence_unlock(DS_1);
   offset+=kk;
  }
  for(int hg=0;hg<groups;hg+=grouped?groups:1)for(int mi=0;mi<mt;++mi){
   int h=hk*groups+hg,m0=mi*p.qt,rows=p.m-m0;if(rows>p.qt)rows=p.qt;if(grouped)rows=groups;
   GM2LM(ranges+((b*p.h+h)*mt+mi)*8,r,32);
   if(grouped){
    for(int g=0;g<groups;++g){
     GM2LM(ranges+((b*p.h+h+g)*mt+mi)*8,r,32);
     _global_ptr_ const unsigned char *src=q+(h+g)*p.q[2]+b*p.q[3];
     xfence_lock(DMAIN_0);dma_cfg(DMA_L2_0|DMA_INT16,DMA_GM|DMA_FP16|highptr(src),r[0]>0?r[0]:1.0f);dma_i((short*)0+g*p.d,lowptr(src),p.d*2);xfence_unlock(DMAIN_0);
    }
    xfence_lock(DS_0);ds_shuffle_cfg_datatype(DS_INT16);ds_shuffle_cfg_blkx(p.d);ds_cfg_output_bank(16);
    for(int j=0;j<(groups+15)/16;++j)ds_shuffle_batch(&((v16i16*)0)[j*p.d][0],(short*)0+j*16*p.d,p.d,DS_L1D);xfence_unlock(DS_0);
    r[0]=1.0f;
   }else load(q+m0*p.q[1]+h*p.q[2]+b*p.q[3],rows,p.d,p.q[1],0,r[0]);
   // L1D is unchanged by MAC/EW/RS/DMAOUT and by the independent L1W bank.
   int width=KQ_OUTPUT_KEYS,capacity=32768/rows;capacity=capacity/32*32;if(width>capacity)width=capacity;
   for(int n0=0;n0<nkeys;n0+=width){
    int cols=nkeys-n0;if(cols>width)cols=width;int vectors=cols/16;
    xfence_lock(MAC);mm_cfg_stride(1);mm_cfg_dequant_scale(a[0]*r[0]*(1.0f/(32767.0f*32767.0f)));mm_cfg_basic_int16(p.d,rows);
    for(int j=0;j<vectors;++j)mm_int16((v16f32*)0+j*rows,(v16i16*)0,(v16i16*)0+(n0/16+j)*p.d);xfence_unlock(MAC);
    v16f32 *e=(v16f32*)0;
    xfence_lock(EW);ew_cfg_activation_type(EW_NOACT);ew_cfg_coeff_scalar(1.0f,0.0f,0.0f);ew_cfg_findmax(0,0.0f);ew_cfg_stream_size(rows*vectors);ew_cfg_col_size(16);ew_dsmadd(e,e,e,EW_L2E);xfence_unlock(EW);
    xfence_lock(RS);rs_row_cfg_stride(1,vectors);rs_row_cfg_loop(rows);for(int j=0;j<vectors;++j)rs_row_batch(&e[j][0],&e[j*rows][0],16);xfence_unlock(RS);
    _global_ptr_ float *out=score+((b*p.h+h)*p.m+m0)*p.block+first+n0;
    xfence_lock(DMAOUT);dma_cfg(DMA_GM|DMA_FP32|highptr(out),DMA_L2R|DMA_FP32,0.0f);dma_cfg_2d(rows,p.block*4,cols*4);dma_o_2d(lowptr(out),e,cols*4);xfence_unlock(DMAOUT);
   }
  }
 }
}
