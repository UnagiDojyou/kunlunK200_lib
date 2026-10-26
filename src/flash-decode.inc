#include "../kernels/flash-decode-config.h"
namespace {
bool flash_decode(FlashState &s,int id,void *y,const void *q,const void *k,const void *v,const void *mask,const KunlunFlashConfig &base){
 if(base.m!=1)return false;
 const char *mode=std::getenv("KUNLUN_FLASH_DECODING");bool automatic=!mode||!std::strcmp(mode,"auto"),forced=mode&&!std::strcmp(mode,"on");
 if(!automatic&&!forced)return false;
 const char *kq_mode=std::getenv("KUNLUN_FLASH_KQ"),*online_mode=std::getenv("KUNLUN_FLASH_ONLINE"),*pv_mode=std::getenv("KUNLUN_FLASH_PV");
 if((kq_mode&&std::strcmp(kq_mode,"auto")&&std::strcmp(kq_mode,"reuse1024"))||(online_mode&&std::strcmp(online_mode,"auto"))||(pv_mode&&!std::strcmp(pv_mode,"narrow")))return false;
 // Only adopt the real-LLM shape/layout with measured end-to-end gains.
 // Nongrouped 7B TG already fills all four SDNNs and did not improve.
 bool measured=base.d==64&&base.h==32&&base.hk==4&&base.b==1&&base.tg_group&&base.block==1024&&base.s>=2048&&base.s<=4352&&base.has_mask&&base.mask_type&&base.max_bias==0&&base.k[1]==base.d*base.hk*2&&base.k[2]==base.d*2&&base.v[1]==base.d*base.hk*2&&base.v[2]==base.d*2;
 bool hbm_requested=std::getenv("KUNLUN_FLASH_HBM")&&!std::strcmp(std::getenv("KUNLUN_FLASH_HBM"),"1");
 if(automatic&&(!measured||hbm_requested))return false;
 bool custom=std::getenv("KUNLUN_FLASH_DECODING_CHUNK")||std::getenv("KUNLUN_FLASH_DECODING_CHUNKS")||std::getenv("KUNLUN_FLASH_DECODING_FIXED_KEYS");
 bool tuned_fixed=measured&&!custom&&base.s>4096;

 int tiles=base.s/32,chunks=(base.s+1023)/1024;
 const char *length=std::getenv("KUNLUN_FLASH_DECODING_CHUNK");
 if(length){int n=std::atoi(length);if(n==256||n==512||n==1024)chunks=(base.s+n-1)/n;}
 int power2=1;while(power2<chunks)power2*=2;chunks=power2;
 const char *number=std::getenv("KUNLUN_FLASH_DECODING_CHUNKS");
 if(number){int n=std::atoi(number);if(n>=1&&n<=64)chunks=n;}
 int fixed=tuned_fixed?1024:0;if(const char *raw=std::getenv("KUNLUN_FLASH_DECODING_FIXED_KEYS")){int n=std::atoi(raw);if(n==32||n==64||n==128||n==256||n==512||n==1024)fixed=n;}
 if(fixed)chunks=(base.s+fixed-1)/fixed;
 if(chunks>tiles)chunks=tiles;
 if(chunks<1||chunks>64)return false;
 FlashDecodeConfig fd{};fd.p=base;fd.chunks=chunks;fd.chunk_keys=fixed;int maxkeys=fixed?std::min(fixed,base.s):((tiles+chunks-1)/chunks)*32;fd.p.block=std::min(base.block,maxkeys);
 size_t rows=(size_t)base.h*base.b*chunks,sc=rows*fd.p.block*4,ou=rows*base.d*4,total=sc+2*ou+rows*32;
 if(total>16*1024*1024)return false;
 if(!s.l3){s.l3=native_l3[id].lock();if(!s.l3){s.l3=std::make_shared<NativeL3>(id);native_l3[id]=s.l3;}}
 bool hbm=std::getenv("KUNLUN_FLASH_HBM")&&!std::strcmp(std::getenv("KUNLUN_FLASH_HBM"),"1");void *workspace=nullptr;
 if(hbm){native_reserve(s.hbm,id,total);workspace=s.hbm->data();}else if(s.l3->reserve(total))workspace=s.l3->pointer;else return false;
 native_reserve(s.config,id,256);if(s.config_cache.size()!=sizeof(fd)||std::memcmp(s.config_cache.data(),&fd,sizeof(fd))){check(xpu_memcpy(s.config->data(),&fd,sizeof(fd),XPU_HOST_TO_DEVICE));check(xpu_wait());s.config_cache.assign((unsigned char*)&fd,(unsigned char*)&fd+sizeof(fd));}
 s.workspace=total;s.last_l3=!hbm;s.last_block=fd.p.block;s.last_qt=base.qt;
 void *score=workspace,*prob=score,*partial=(char*)score+sc,*acc=(char*)partial+ou,*stats=(char*)acc+ou,*cp=s.config->data(),*kp=s.peaks->data(),*qr=s.ranges->data(),*vp=(char*)kp+256;
 bool profile=std::getenv("KUNLUN_FLASH_DECODING_PROFILE")&&base.h*base.b>=32;
 bool head=!std::getenv("KUNLUN_FLASH_DECODING_LAYOUT")||std::strcmp(std::getenv("KUNLUN_FLASH_DECODING_LAYOUT"),"chunk");
 if(profile)head=true;
 const char *work=std::getenv("KUNLUN_FLASH_DECODING_WORK");bool affinity=head&&((!work||!std::strcmp(work,"auto"))?tuned_fixed:!std::strcmp(work,"head"));
 auto &online=profile?s.profile_decode_online:(head?s.decode_online_head:s.decode_online);auto &pv=profile?(affinity?s.profile_decode_pv_head:s.profile_decode_pv):(head?(affinity?s.decode_pv_head_affinity:s.decode_pv_head):s.decode_pv);auto &reduce=profile?s.profile_decode_reduce:(head?s.decode_reduce_head:s.decode_reduce);
 auto &kq_program=profile?(affinity?s.profile_decode_kq_head:s.profile_decode_kq):(affinity?s.decode_kq_head:s.decode_kq);load_kernel(kq_program,profile?(affinity?"kunlun-flash-profile-decode-kq-head":"kunlun-flash-profile-decode-kq"):(affinity?"kunlun-flash-decode-kq-head":"kunlun-flash-decode-kq"),id,true);load_kernel(pv,profile?(affinity?"kunlun-flash-profile-decode-pv-head":"kunlun-flash-profile-decode-pv"):(head?(affinity?"kunlun-flash-decode-pv-head-affinity":"kunlun-flash-decode-pv-head"):"kunlun-flash-decode-pv"),id,true);load_kernel(online,profile?"kunlun-flash-profile-decode-online":(head?"kunlun-flash-decode-online-head":"kunlun-flash-decode-online"),id,false);load_kernel(reduce,profile?"kunlun-flash-profile-decode-reduce":(head?"kunlun-flash-decode-reduce-head":"kunlun-flash-decode-reduce"),id,false);
 bool trace=std::getenv("KUNLUN_FLASH_TRACE")||profile;auto tick=std::chrono::steady_clock::now();
 auto stage=[&](const char *name){if(trace){check(xpu_wait());auto now=std::chrono::steady_clock::now();std::fprintf(stderr,"DECODE_STAGE stage=%s ms=%.6f\n",name,std::chrono::duration<double,std::milli>(now-tick).count());tick=now;if(profile){flash_profile_report(name,s,stats);tick=std::chrono::steady_clock::now();}}};
 int waves=(maxkeys+fd.p.block-1)/fd.p.block;
 for(int step=0;step<waves;++step){
  native_launch(kq_program,{native_arg(score),native_arg(k),native_arg(q),native_arg(kp),native_arg(qr),native_arg(cp),native_arg(step)},8);stage("kq");
  native_launch(online,{native_arg(y),native_arg(prob),native_arg(acc),native_arg(stats),native_arg(partial),native_arg(score),native_arg(mask),native_arg(qr),native_arg(cp),native_arg(step)},16);stage("online");
  native_launch(pv,{native_arg(partial),native_arg(v),native_arg(prob),native_arg(vp),native_arg(cp),native_arg(step)},8);stage("pv");
 }
 native_launch(reduce,{native_arg(y),native_arg(acc),native_arg(partial),native_arg(stats),native_arg(cp)},16);check(xpu_wait());stage("reduce");
 ++s.calls;if(!s.decode_calls++||trace)std::fprintf(stderr,"KUNLUN Flash-Decoding PD%d D=%d S=%d H=%d HK=%d chunks=%d fixed=%d block=%d waves=%d workspace=%zu storage=%s pool=%zu layout=%s work=%s calls=%zu\n",id,base.d,base.s,base.h,base.hk,chunks,fixed,fd.p.block,waves,total,hbm?"HBM":"L3",hbm?0:s.l3->bytes,head?"head":"chunk",affinity?"head":"chunk",s.calls);
 return true;
}
}
