#include "../kernels/flash-config.h"
namespace {
struct FlashState {
    int id;
    Kernel::Impl kq,kq1024,pv,pv_wide,online,online_ordered,qpack,kpack,vpack,range;
    std::shared_ptr<NativeL3> l3;
    std::unique_ptr<Buffer> config,peaks,ranges,hbm;
    std::vector<unsigned char> config_cache;
    xdnn::Context *vendor=nullptr;
    size_t workspace=0,calls=0;
    bool last_l3=false;
    int last_block=0,last_qt=0;
    explicit FlashState(int pd):id(pd){}
    ~FlashState(){
        std::lock_guard<std::recursive_mutex> lock(mutex);select(id);check(xpu_wait());
        if(vendor)xdnn::destroy_context(vendor);
        for(auto *s:{&kq,&kq1024,&pv,&pv_wide,&online,&online_ordered,&qpack,&kpack,&vpack,&range}){
            if(s->vendor)xdnn::destroy_context(s->vendor);
            release(id,s->quant,s->quant_size);release(id,s->scratch,s->scratch_size);release(id,s->alias_input,s->alias_input_size);
            if(s->code)check(xpu_free(s->code));
        }
    }
};
int flash_query_tile(){
    const char *s=std::getenv("KUNLUN_FLASH_QUERY_TILE");
    if(!s)return 128;
    if(s&&!std::strcmp(s,"16"))return 16;
    if(s&&!std::strcmp(s,"64"))return 64;
    if(s&&!std::strcmp(s,"128"))return 128;
    return 32;
}
int flash_block(int queries=128,int groups=1){
    const char *s=std::getenv("KUNLUN_FLASH_BLOCK");if(!s)return queries==1&&groups<=64?1024:512;
    if(!std::strcmp(s,"128"))return 128;
    if(!std::strcmp(s,"512"))return 512;
    if(!std::strcmp(s,"1024"))return 1024;
    return 256;
}
}
const char *flash_attention_reason(const Flash4Layout &p){
    for(auto *t:{&p.q,&p.k,&p.v,&p.y})if(*tensor4_reason(*t))return tensor4_reason(*t);
    if(p.k.type!=InputType::f16 || p.v.type!=InputType::f16 || p.y.type!=InputType::f32)return "flash types";
    int d=p.q.ne[0],m=p.q.ne[1],h=p.q.ne[2],b=p.q.ne[3];
    if((d!=64&&d!=128) || m>256 || h>128 || b>16 || p.k.ne[1]%32 || p.k.ne[1]>16384)return "flash dimensions";
    if(p.k.ne[0]!=d || p.v.ne[0]!=d || p.k.ne[1]!=p.v.ne[1] || p.k.ne[2]!=p.v.ne[2] || h%p.k.ne[2] || b%p.k.ne[3] || b%p.v.ne[3])return "flash KV broadcast";
    if(p.y.ne[0]!=d || p.y.ne[1]!=h || p.y.ne[2]!=m || p.y.ne[3]!=b)return "flash output";
    if(!std::isfinite(p.scale) || !std::isfinite(p.max_bias) || p.max_bias<0)return "flash scale";
    if(p.has_mask){if(*tensor4_reason(p.mask))return "flash mask layout";if(p.mask.ne[0]!=p.k.ne[1] || p.mask.ne[1]<m || h%p.mask.ne[2] || b%p.mask.ne[3])return "flash mask broadcast";}
    size_t rows=(size_t)m*h*b,bytes=rows*(4*std::min(flash_block(p.q.ne[1],p.q.ne[2]/p.k.ne[2]),p.k.ne[1])+8*d+32);
    if((p.q.ne[1]==1?p.q.ne[2]/p.k.ne[2]:std::min(flash_query_tile(),p.q.ne[1]))*std::min(flash_block(p.q.ne[1],p.q.ne[2]/p.k.ne[2]),p.k.ne[1])*2>128*1024)return "flash input SRAM";
    if(bytes>16*1024*1024)return "flash workspace";
    return "";
}
bool flash_attention(Kernel &kernel,int id,void *y,size_t yb,const void *q,size_t qb,const void *k,size_t kb,const void *v,size_t vb,const void *mask,size_t mb,const Flash4Layout &p){
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(*flash_attention_reason(p) || id<0 || id>1 || (kernel.impl_->id!=-1 && kernel.impl_->id!=id))return false;
    if(!tensor4_pointer(y,yb,p.y)||!tensor4_pointer(q,qb,p.q)||!tensor4_pointer(k,kb,p.k)||!tensor4_pointer(v,vb,p.v)||(p.has_mask&&!tensor4_pointer(mask,mb,p.mask)))return false;
    if((((uintptr_t)q|(uintptr_t)k|(uintptr_t)v)&31)!=0)return false;
    if(tensor4_overlap(y,p.y,q,p.q)||tensor4_overlap(y,p.y,k,p.k)||tensor4_overlap(y,p.y,v,p.v))return false;
    if(p.has_mask&&tensor4_overlap(y,p.y,mask,p.mask))return false;
    kernel.impl_->id=id;select(id);
    if(!kernel.impl_->flash)kernel.impl_->flash=std::make_shared<FlashState>(id);
    auto &s=*std::static_pointer_cast<FlashState>(kernel.impl_->flash);
    bool trace=std::getenv("KUNLUN_FLASH_TRACE")!=nullptr;auto start=std::chrono::steady_clock::now(),tick=start;
    auto stage=[&](const char *name){if(trace){check(xpu_wait());auto now=std::chrono::steady_clock::now();std::fprintf(stderr,"FLASH_STAGE stage=%s ms=%.6f\n",name,std::chrono::duration<double,std::milli>(now-tick).count());tick=now;}};
    auto pq=p.q,pk=p.k,pv=p.v;
    // Each owner survives all enqueued blocks. Recompute on every call for KV updates.
    auto half=[&](Kernel::Impl &state,const void *src,Tensor4Layout &layout){
        bool dense=true;int stride=tensor4_bytes(layout);for(int d=0;d<4;++d){if(layout.ne[d]>1 && layout.nb[d]!=stride)dense=false;stride*=layout.ne[d];}
        size_t count=tensor4_count(layout);const void *input=src;
        if(layout.type==InputType::f16 && tensor4_span(layout)==count*2)return src;
        if(!dense){size_t size=count*tensor4_bytes(layout);if(state.scratch_size<size){release(id,state.scratch,state.scratch_size);state.scratch=allocate(id,size);state.scratch_size=size;require(state.scratch);}
            auto flat=layout;int stride=tensor4_bytes(layout);for(int d=0;d<4;++d){flat.nb[d]=stride;stride*=flat.ne[d];}
            auto cfg=attention_config(flat,layout);attention_launch(state,id,"kunlun-attention-copy",cfg,{state.scratch,src});input=state.scratch;
        }
        if(layout.type==InputType::f32){if(state.quant_size<count*2){release(id,state.quant,state.quant_size);state.quant=allocate(id,count*2);state.quant_size=count*2;require(state.quant);}
            if(!s.vendor){s.vendor=xdnn::create_context();require(s.vendor);s.vendor->set_ncluster(4);}
            check(xdnn::cast<float,float16>(s.vendor,(const float*)input,(float16*)state.quant,(int)count));input=state.quant;
        }
        layout.type=InputType::f16;stride=2;for(int d=0;d<4;++d){layout.nb[d]=stride;stride*=layout.ne[d];}return input;
    };
    q=half(s.qpack,q,pq);k=half(s.kpack,k,pk);v=half(s.vpack,v,pv);stage("pack_cast");
    if(!s.vendor){s.vendor=xdnn::create_context();require(s.vendor);s.vendor->set_ncluster(4);}
    native_reserve(s.peaks,id,1024);native_reserve(s.ranges,id,(size_t)((p.q.ne[1]+flash_query_tile()-1)/flash_query_tile())*p.q.ne[2]*p.q.ne[3]*32);
    check(xdnn::findmax<float16>(s.vendor,(const float16*)k,(float*)s.peaks->data(),(int)tensor4_count(pk)));
    void *vmax=(char*)s.peaks->data()+256;
    check(xdnn::findmax<float16>(s.vendor,(const float16*)v,(float*)vmax,(int)tensor4_count(pv)));
    auto flat=pq;auto rcfg=attention_config(flat,pq);rcfg.k=flash_query_tile();
    attention_launch(s.range,id,"kunlun-flash-range",rcfg,{s.ranges->data(),q},true);stage("ranges");
    KunlunFlashConfig cfg{};cfg.d=pq.ne[0];cfg.s=pk.ne[1];cfg.m=pq.ne[1];cfg.h=pq.ne[2];cfg.hk=pk.ne[2];cfg.b=pq.ne[3];cfg.bk=pk.ne[3];cfg.bv=pv.ne[3];cfg.block=std::min(flash_block(cfg.m,cfg.h/cfg.hk),cfg.s);cfg.qt=flash_query_tile();
    std::copy(pq.nb,pq.nb+4,cfg.q);std::copy(pk.nb,pk.nb+4,cfg.k);std::copy(pv.nb,pv.nb+4,cfg.v);std::copy(p.y.nb,p.y.nb+4,cfg.y);
    if(p.has_mask){std::copy(p.mask.nb,p.mask.nb+4,cfg.mask);std::copy(p.mask.ne,p.mask.ne+4,cfg.mask_ne);cfg.mask_type=p.mask.type==InputType::f16;}else for(int d=0;d<4;++d)cfg.mask_ne[d]=1;cfg.tg_group=!std::getenv("KUNLUN_FLASH_TG_GROUP")||std::strcmp(std::getenv("KUNLUN_FLASH_TG_GROUP"),"0");cfg.has_mask=p.has_mask;cfg.scale=p.scale;cfg.max_bias=p.max_bias;
    native_reserve(s.config,id,256);if(s.config_cache.size()!=sizeof(cfg)||std::memcmp(s.config_cache.data(),&cfg,sizeof(cfg))){check(xpu_memcpy(s.config->data(),&cfg,sizeof(cfg),XPU_HOST_TO_DEVICE));check(xpu_wait());s.config_cache.assign((unsigned char*)&cfg,(unsigned char*)&cfg+sizeof(cfg));}
    size_t rows=(size_t)cfg.m*cfg.h*cfg.b,sc=rows*cfg.block*4,pr=0,ou=rows*cfg.d*4,total=sc+pr+2*ou+rows*32;
    if(!s.l3){s.l3=native_l3[id].lock();if(!s.l3){s.l3=std::make_shared<NativeL3>(id);native_l3[id]=s.l3;}}
    void *workspace=nullptr;bool hbm=std::getenv("KUNLUN_FLASH_HBM")&&!std::strcmp(std::getenv("KUNLUN_FLASH_HBM"),"1");
    if(!hbm&&s.l3->reserve(total))workspace=s.l3->pointer;
    else{native_reserve(s.hbm,id,total);workspace=s.hbm->data();hbm=true;}
    s.workspace=total;s.last_l3=!hbm;s.last_block=cfg.block;s.last_qt=cfg.qt;
    void *score=workspace,*prob=score,*partial=(char*)score+sc,*acc=(char*)partial+ou,*stats=(char*)acc+ou,*cp=s.config->data(),*kp=s.peaks->data(),*qr=s.ranges->data();
    const char *kq_mode=std::getenv("KUNLUN_FLASH_KQ");
    int key_tiles=cfg.block/32,partitions=(4+cfg.b*cfg.hk-1)/(cfg.b*cfg.hk);
    if(partitions>key_tiles)partitions=key_tiles;
    // With only one key tile per SDNN part there is no Q reuse to amortize.
    bool automatic=!kq_mode||!std::strcmp(kq_mode,"auto");
    bool kq_optimized=(automatic&&key_tiles>partitions)||(kq_mode&&!std::strcmp(kq_mode,"reuse1024"));
    auto *kq_program=kq_optimized?&s.kq1024:&s.kq;
    load_kernel(*kq_program,kq_optimized?"kunlun-flash-kq-reuse1024":"kunlun-flash-kq",id,true);bool wide=!std::getenv("KUNLUN_FLASH_PV")||std::strcmp(std::getenv("KUNLUN_FLASH_PV"),"narrow");auto &pv_program=wide?s.pv_wide:s.pv;load_kernel(pv_program,wide?"kunlun-flash-pv-wide":"kunlun-flash-pv",id,true);const char *mode=std::getenv("KUNLUN_FLASH_ONLINE");
    // Ordered FP32 sum preserves the 0.7.0 probability and running-state results.
    // Unknown settings select the conservative reference implementation.
    bool optimized=!mode||!std::strcmp(mode,"auto")||!std::strcmp(mode,"ordered");
    auto &online=optimized?s.online_ordered:s.online;
    load_kernel(online,optimized?"kunlun-flash-online-ordered":"kunlun-flash-online",id,false);
    stage("workspace");
    for(int s0=0;s0<cfg.s;s0+=cfg.block){int count=std::min(cfg.block,cfg.s-s0);
        native_launch(*kq_program,{native_arg(score),native_arg(k),native_arg(q),native_arg(kp),native_arg(qr),native_arg(cp),native_arg(s0),native_arg(count)},8);stage("kq");
        const void *mp=p.has_mask?mask:nullptr;
        native_launch(online,{native_arg(y),native_arg(prob),native_arg(acc),native_arg(stats),native_arg(partial),native_arg(score),native_arg(mp),native_arg(qr),native_arg(cp),native_arg(s0),native_arg(count)},16);stage("online");
        native_launch(pv_program,{native_arg(partial),native_arg(v),native_arg(prob),native_arg(vmax),native_arg(cp),native_arg(s0),native_arg(count)},8);stage("pv");
    }
    int s0=cfg.s,count=0;const void *mp=nullptr;
    native_launch(online,{native_arg(y),native_arg(prob),native_arg(acc),native_arg(stats),native_arg(partial),native_arg(score),native_arg(mp),native_arg(qr),native_arg(cp),native_arg(s0),native_arg(count)},16);check(xpu_wait());stage("final");
    if(!s.calls++||trace)std::fprintf(stderr,"KUNLUN Flash PD%d D=%d S=%d M=%d H=%d block=%d workspace=%zu storage=%s kq=%s calls=%zu\n",id,cfg.d,cfg.s,cfg.m,cfg.h,cfg.block,total,hbm?"HBM":"L3",kq_optimized?"reuse1024":"legacy",s.calls);
    return true;
}

FlashStatistics flash_statistics(const Kernel &kernel){
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!kernel.impl_->flash)return {};
    auto &s=*std::static_pointer_cast<FlashState>(kernel.impl_->flash);size_t bytes=0;
    for(auto *p:{&s.qpack,&s.kpack,&s.vpack,&s.range})bytes+=p->scratch_size+p->quant_size+p->alias_input_size;
    for(auto *p:{s.config.get(),s.peaks.get(),s.ranges.get(),s.hbm.get()})if(p)bytes+=p->size();
    return {s.calls,bytes,s.last_l3?s.workspace:0,s.last_block,s.last_qt};
}
