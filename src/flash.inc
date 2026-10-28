#include "../kernels/flash-config.h"
namespace {
struct FlashState {
    int id;
    Kernel::Impl kv_dma_kq,kv_dma_pv,kv_dma_decode_kq,kv_dma_decode_pv,kv_dma_decode_pv_head,pipeline_profile_kq,pipeline_profile_online,pipeline_profile_pv,kq,kq1024,pv,pv_wide,online,online_ordered,online_unroll16,online_simd8,decode_online_simd8,decode_online_head_simd8,decode_kq,decode_kq_head,decode_pv,decode_online,decode_reduce,decode_pv_head,decode_pv_head_affinity,decode_online_head,decode_reduce_head,profile_kq,profile_pv,profile_online,profile_decode_kq,profile_decode_kq_head,profile_decode_pv,profile_decode_pv_head,profile_decode_online,profile_decode_reduce,qpack,kpack,vpack,range;
    std::shared_ptr<NativeL3> l3;
    std::unique_ptr<Buffer> config,peaks,ranges,hbm,pipeline_score,pipeline_timeline;
    XPUStream pipeline_sd=nullptr,pipeline_cl=nullptr;
    std::vector<unsigned char> config_cache;
    xdnn::Context *vendor=nullptr;
    size_t workspace=0,calls=0,decode_calls=0,pipeline_calls=0,pipeline_blocks=0;
    bool last_l3=false;
    int last_block=0,last_qt=0;
    explicit FlashState(int pd):id(pd){}
    ~FlashState(){
        std::lock_guard<std::recursive_mutex> lock(mutex);select(id);check(xpu_wait());
        if(pipeline_sd){check(xpu_wait(pipeline_sd));check(xpu_stream_destroy(pipeline_sd));}
        if(pipeline_cl){check(xpu_wait(pipeline_cl));check(xpu_stream_destroy(pipeline_cl));}
        if(std::getenv("KUNLUN_FLASH_PIPELINE_STATS")&&calls)std::fprintf(stderr,"FLASH_PIPELINE_STATS pd=%d calls=%zu pipeline_calls=%zu blocks=%zu\n",id,calls,pipeline_calls,pipeline_blocks);
        if(vendor)xdnn::destroy_context(vendor);
        for(auto *s:{&kv_dma_kq,&kv_dma_pv,&kv_dma_decode_kq,&kv_dma_decode_pv,&kv_dma_decode_pv_head,&pipeline_profile_kq,&pipeline_profile_online,&pipeline_profile_pv,&kq,&kq1024,&pv,&pv_wide,&online,&online_ordered,&online_unroll16,&online_simd8,&decode_online_simd8,&decode_online_head_simd8,&decode_kq,&decode_kq_head,&decode_pv,&decode_online,&decode_reduce,&decode_pv_head,&decode_pv_head_affinity,&decode_online_head,&decode_reduce_head,&profile_kq,&profile_pv,&profile_online,&profile_decode_kq,&profile_decode_kq_head,&profile_decode_pv,&profile_decode_pv_head,&profile_decode_online,&profile_decode_reduce,&qpack,&kpack,&vpack,&range}){
            if(s->vendor)xdnn::destroy_context(s->vendor);
            release(id,s->quant,s->quant_size);release(id,s->scratch,s->scratch_size);release(id,s->alias_input,s->alias_input_size);
            if(s->code)check(xpu_free(s->code));
        }
    }
};
// A missing optional image or code-buffer OOM leaves the old program usable.
bool flash_kv_load(Kernel::Impl &program,const char *name,int id){
 if(program.fn)return true;select(id);std::string path=kernel_directory()+"/"+name;
 std::ifstream bin(path+".bin",std::ios::binary),entry(path+".entry");std::vector<char> bytes((std::istreambuf_iterator<char>(bin)),{});std::string symbol;std::getline(entry,symbol);
 if(bytes.empty()||symbol.empty())return false;bytes.resize(((bytes.size()+255)&~size_t(255))+256,0);if(bytes.size()>16384)return false;
 void *code=nullptr;int rc=xpu_malloc(&code,bytes.size());if(rc){if(rc==XPUERR_NOMEM||rc==-XPUERR_NOMEM||rc==XPUERR_NOSUPPORT||rc==-XPUERR_NOSUPPORT)return false;check(rc);}
 check(xpu_memcpy(code,bytes.data(),bytes.size(),XPU_HOST_TO_DEVICE));check(xpu_wait());program.id=id;program.code=code;program.symbol=std::move(symbol);check(xpu_create_sd_func(&program.fn,(uint64_t)(uintptr_t)code,bytes.size(),0,0,0,program.symbol.c_str(),true));return true;
}
// The extra scalar changes scheduling/layout only; all old binaries remain available.
void flash_kv_launch(Kernel::Impl &program,std::initializer_list<NativeArgument> args,int group,XPUStream stream=nullptr,const void *timeline=nullptr,int slot=0){
 if(!group){native_launch_stream(program,args,8,stream,timeline,slot);return;}
 require(!timeline);size_t off=0;for(auto a:args){check(xpu_launch_argument_set(a.value,a.size,off));off+=a.size;}check(xpu_launch_argument_set(&group,sizeof(group),off));check(xpu_launch_config(4,8,stream));check(xpu_launch_async(program.fn));
}
int flash_kv_group(const KunlunFlashConfig &p,const int *stride,int chunks=1,bool decoding=false){
 const char *mode=std::getenv("KUNLUN_FLASH_KV_DMA");int g=0;
 if(stride[0]!=2||stride[2]!=p.d*2||stride[1]!=p.d*p.hk*2)return 0;
 bool automatic=!mode||!std::strcmp(mode,"auto");
 if(automatic){
  // Real 7B TG only. PP and the small-KV-head experiments remain opt-in.
  bool measured=!decoding&&p.m==1&&p.d==128&&p.hk==32&&p.h==32&&p.b==1&&p.qt==128&&chunks==1&&p.s>=512&&p.s<=4352&&p.s%256==0&&p.block==std::min(1024,p.s)&&p.has_mask&&p.mask_type&&p.max_bias==0&&p.k[0]==2&&p.k[1]==p.d*p.hk*2&&p.k[2]==p.d*2&&p.v[0]==2&&p.v[1]==p.d*p.hk*2&&p.v[2]==p.d*2;
  if(measured)g=8;
 }else if(!std::strcmp(mode,"coalesced"))g=p.hk>=32?8:0;
 else if(!std::strcmp(mode,"decode2"))g=p.m==1&&p.hk==4&&chunks>=4?2:0;
 else if(!std::strcmp(mode,"decode4"))g=p.m==1&&p.hk==4&&chunks>=4?4:0;
 else if(!std::strcmp(mode,"group2"))g=2;
 else if(!std::strcmp(mode,"group4"))g=4;
 else if(!std::strcmp(mode,"group8"))g=8;
 if(g<1||p.hk%g||(size_t)p.block*p.d*g*2>2*1024*1024)return 0;
 return g;
}
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
#include "flash-profile.inc"
#include "flash-decode.inc"
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
    bool profile=std::getenv("KUNLUN_FLASH_DECODING_PROFILE")&&p.q.ne[1]==1&&p.q.ne[2]*p.q.ne[3]>=32;
    bool trace=std::getenv("KUNLUN_FLASH_TRACE")!=nullptr||profile;auto start=std::chrono::steady_clock::now(),tick=start;
    auto stage=[&](const char *name){if(trace){check(xpu_wait());auto now=std::chrono::steady_clock::now();std::fprintf(stderr,"FLASH_STAGE stage=%s ms=%.6f kv_max=%s phase=%s\n",name,std::chrono::duration<double,std::milli>(now-tick).count(),std::getenv("KUNLUN_FLASH_KV_MAX")?std::getenv("KUNLUN_FLASH_KV_MAX"):"default",std::getenv("K200_PROFILE_PHASE")?std::getenv("K200_PROFILE_PHASE"):"none");tick=now;if(profile){flash_profile_report(name,s);tick=std::chrono::steady_clock::now();}}};
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
    void *vmax=(char*)s.peaks->data()+256;
    bool max_cached=kv_max_prepare(id,k,pk,v,pv,p.q.ne[1],s.peaks->data(),s.vendor);stage("kv_max_metadata");
    if(!max_cached){
        check(xdnn::findmax<float16>(s.vendor,(const float16*)k,(float*)s.peaks->data(),(int)tensor4_count(pk)));stage("k_findmax");
        check(xdnn::findmax<float16>(s.vendor,(const float16*)v,(float*)vmax,(int)tensor4_count(pv)));stage("v_findmax");
        kv_max_finish(id,k,pk,v,pv,s.peaks->data());stage("kv_max_seed");
    }
    auto flat=pq;auto rcfg=attention_config(flat,pq);rcfg.k=flash_query_tile();
    attention_launch(s.range,id,"kunlun-flash-range",rcfg,{s.ranges->data(),q},true);stage("ranges");
    KunlunFlashConfig cfg{};cfg.d=pq.ne[0];cfg.s=pk.ne[1];cfg.m=pq.ne[1];cfg.h=pq.ne[2];cfg.hk=pk.ne[2];cfg.b=pq.ne[3];cfg.bk=pk.ne[3];cfg.bv=pv.ne[3];cfg.block=std::min(flash_block(cfg.m,cfg.h/cfg.hk),cfg.s);cfg.qt=flash_query_tile();
    std::copy(pq.nb,pq.nb+4,cfg.q);std::copy(pk.nb,pk.nb+4,cfg.k);std::copy(pv.nb,pv.nb+4,cfg.v);std::copy(p.y.nb,p.y.nb+4,cfg.y);
    if(p.has_mask){std::copy(p.mask.nb,p.mask.nb+4,cfg.mask);std::copy(p.mask.ne,p.mask.ne+4,cfg.mask_ne);cfg.mask_type=p.mask.type==InputType::f16;}else for(int d=0;d<4;++d)cfg.mask_ne[d]=1;cfg.tg_group=!std::getenv("KUNLUN_FLASH_TG_GROUP")||std::strcmp(std::getenv("KUNLUN_FLASH_TG_GROUP"),"0");cfg.has_mask=p.has_mask;cfg.scale=p.scale;cfg.max_bias=p.max_bias;
    if(flash_decode(s,id,y,q,k,v,p.has_mask?mask:nullptr,cfg))return true;
    const char *pipeline_mode=std::getenv("KUNLUN_FLASH_PIPELINE");
    const char *pipeline_buffer=std::getenv("KUNLUN_FLASH_PIPELINE_BUFFER");
    const char *pipeline_kq=std::getenv("KUNLUN_FLASH_KQ"),*pipeline_online=std::getenv("KUNLUN_FLASH_ONLINE"),*pipeline_pv=std::getenv("KUNLUN_FLASH_PV");
    bool pipeline_auto=!pipeline_mode||!std::strcmp(pipeline_mode,"auto");
    bool pipeline_defaults=(!pipeline_kq||!std::strcmp(pipeline_kq,"auto")||!std::strcmp(pipeline_kq,"reuse1024"))&&(!pipeline_online||!std::strcmp(pipeline_online,"auto")||!std::strcmp(pipeline_online,"ordered")||!std::strcmp(pipeline_online,"unroll16")||!std::strcmp(pipeline_online,"simd8"))&&(!pipeline_pv||std::strcmp(pipeline_pv,"narrow"))&&(!pipeline_buffer||!std::strcmp(pipeline_buffer,"auto")||!std::strcmp(pipeline_buffer,"l3"));
    // Limit auto to the TG families with measured LLM benefits on both PDs.
    bool pipeline_shape=cfg.m==1&&cfg.h==32&&cfg.b==1&&cfg.block==1024&&cfg.qt==128&&((cfg.d==128&&cfg.hk==32&&cfg.s>1024&&cfg.s<=4352)||(cfg.d==64&&cfg.hk==4&&cfg.s>=8192&&cfg.s<=8448));
    bool pipeline=((pipeline_auto&&pipeline_shape&&pipeline_defaults)||(pipeline_mode&&!std::strcmp(pipeline_mode,"on")))&&cfg.s>cfg.block&&!trace;
    if(pipeline&&!s.pipeline_sd){
        int sd_rc=xpu_stream_create(&s.pipeline_sd),cl_rc=sd_rc?sd_rc:xpu_stream_create(&s.pipeline_cl);
        if(sd_rc||cl_rc){if(s.pipeline_sd)check(xpu_stream_destroy(s.pipeline_sd));s.pipeline_sd=nullptr;s.pipeline_cl=nullptr;pipeline=false;}
    }
    native_reserve(s.config,id,256);if(s.config_cache.size()!=sizeof(cfg)||std::memcmp(s.config_cache.data(),&cfg,sizeof(cfg))){check(xpu_memcpy(s.config->data(),&cfg,sizeof(cfg),XPU_HOST_TO_DEVICE));check(xpu_wait());s.config_cache.assign((unsigned char*)&cfg,(unsigned char*)&cfg+sizeof(cfg));}
    size_t rows=(size_t)cfg.m*cfg.h*cfg.b,sc=rows*cfg.block*4,pr=0,ou=rows*cfg.d*4,total=sc+pr+2*ou+rows*32;
    if(!s.l3){s.l3=native_l3[id].lock();if(!s.l3){s.l3=std::make_shared<NativeL3>(id);native_l3[id]=s.l3;}}
    void *workspace=nullptr,*next_score=nullptr,*state_workspace=nullptr;
    bool hbm=std::getenv("KUNLUN_FLASH_HBM")&&!std::strcmp(std::getenv("KUNLUN_FLASH_HBM"),"1");
    const char *buffer=std::getenv("KUNLUN_FLASH_PIPELINE_BUFFER");
    bool state_hbm=pipeline&&buffer&&!std::strcmp(buffer,"state-hbm")&&2*sc<=16*1024*1024&&!hbm;
    bool both_l3=pipeline&&!hbm&&total+sc<=16*1024*1024&&(!buffer||std::strcmp(buffer,"hbm"));
    if(pipeline&&buffer&&!std::strcmp(buffer,"l3")&&!both_l3)pipeline=false;
    size_t requested=state_hbm?2*sc:total+(both_l3?sc:0);
    if(!hbm&&s.l3->reserve(requested)){
        workspace=s.l3->pointer;
        if(state_hbm){native_reserve(s.pipeline_score,id,2*ou+rows*32);state_workspace=s.pipeline_score->data();next_score=(char*)workspace+sc;total=2*sc;}
        else if(both_l3){next_score=(char*)workspace+total;total+=sc;}
    }else{
        // A failed reserve releases the previous pool. Never retain that pointer.
        native_reserve(s.hbm,id,total);workspace=s.hbm->data();hbm=true;state_hbm=false;both_l3=false;
        if(buffer&&!std::strcmp(buffer,"l3"))pipeline=false;
    }
    if(pipeline&&pipeline_auto&&hbm)pipeline=false;
    if(pipeline&&!next_score){native_reserve(s.pipeline_score,id,sc);next_score=s.pipeline_score->data();}
    s.workspace=total;s.last_l3=!hbm;s.last_block=cfg.block;s.last_qt=cfg.qt;
    void *score=workspace,*prob=score,*partial=state_workspace?state_workspace:(char*)score+sc,*acc=(char*)partial+ou,*stats=(char*)acc+ou,*cp=s.config->data(),*kp=s.peaks->data(),*qr=s.ranges->data();
    const char *kq_mode=std::getenv("KUNLUN_FLASH_KQ");
    int key_tiles=cfg.block/32,partitions=(4+cfg.b*cfg.hk-1)/(cfg.b*cfg.hk);
    if(partitions>key_tiles)partitions=key_tiles;
    // With only one key tile per SDNN part there is no Q reuse to amortize.
    bool automatic=!kq_mode||!std::strcmp(kq_mode,"auto");
    bool kq_optimized=(automatic&&key_tiles>partitions)||(kq_mode&&!std::strcmp(kq_mode,"reuse1024"));
    if(profile&&!kq_optimized)profile=false;
    auto *kq_program=profile?&s.profile_kq:(kq_optimized?&s.kq1024:&s.kq);
    load_kernel(*kq_program,profile?"kunlun-flash-profile-kq":(kq_optimized?"kunlun-flash-kq-reuse1024":"kunlun-flash-kq"),id,true);bool wide=!std::getenv("KUNLUN_FLASH_PV")||std::strcmp(std::getenv("KUNLUN_FLASH_PV"),"narrow");auto &pv_program=profile?s.profile_pv:(wide?s.pv_wide:s.pv);load_kernel(pv_program,profile?"kunlun-flash-profile-pv":(wide?"kunlun-flash-pv-wide":"kunlun-flash-pv"),id,true);const char *mode=std::getenv("KUNLUN_FLASH_ONLINE");
    int kv_kg=!profile&&kq_optimized?flash_kv_group(cfg,cfg.k):0;
    int kv_vg=!profile&&wide?flash_kv_group(cfg,cfg.v):0;
    if(kv_kg){if(flash_kv_load(s.kv_dma_kq,"kunlun-flash-kq-kv-group",id))kq_program=&s.kv_dma_kq;else kv_kg=0;}
    if(kv_vg&&!flash_kv_load(s.kv_dma_pv,"kunlun-flash-pv-kv-group",id))kv_vg=0;auto &selected_pv=kv_vg?s.kv_dma_pv:pv_program;
    if(std::getenv("KUNLUN_FLASH_KV_DMA_TRACE"))std::fprintf(stderr,"FLASH_KV_DMA pd=%d D=%d S=%d M=%d HK=%d Kstride=%d,%d,%d Vstride=%d,%d,%d kmod4096=%zu vmod4096=%zu kg=%d vg=%d\n",id,cfg.d,cfg.s,cfg.m,cfg.hk,cfg.k[0],cfg.k[1],cfg.k[2],cfg.v[0],cfg.v[1],cfg.v[2],(uintptr_t)k%4096,(uintptr_t)v%4096,kv_kg,kv_vg);
    // Ordered FP32 sum preserves the 0.7.0 probability and running-state results.
    // Unknown settings select the conservative reference implementation.
    bool simd8=mode&&!std::strcmp(mode,"simd8");
    bool unroll16=mode&&!std::strcmp(mode,"unroll16");
    bool optimized=!mode||!std::strcmp(mode,"auto")||!std::strcmp(mode,"ordered");
    auto &online=profile?s.profile_online:(simd8?s.online_simd8:(unroll16?s.online_unroll16:(optimized?s.online_ordered:s.online)));
    load_kernel(online,profile?"kunlun-flash-profile-online":(simd8?"kunlun-flash-online-simd8":(unroll16?"kunlun-flash-online-unroll16":(optimized?"kunlun-flash-online-ordered":"kunlun-flash-online"))),id,false);
    stage("workspace");
    bool pipeline_profile=pipeline&&!kv_kg&&!kv_vg&&std::getenv("KUNLUN_FLASH_PIPELINE_PROFILE")&&kq_optimized&&wide&&optimized;
    void *timeline=nullptr;
    if(pipeline_profile){
        native_reserve(s.pipeline_timeline,id,((cfg.s+cfg.block-1)/cfg.block*3+1)*4096);
        timeline=s.pipeline_timeline->data();
        load_kernel(s.pipeline_profile_kq,"kunlun-flash-pipeline-profile-kq",id,true);
        load_kernel(s.pipeline_profile_online,"kunlun-flash-pipeline-profile-online",id,false);
        load_kernel(s.pipeline_profile_pv,"kunlun-flash-pipeline-profile-pv",id,true);
    }
    auto &pipe_kq=pipeline_profile?s.pipeline_profile_kq:*kq_program;
    auto &pipe_online=pipeline_profile?s.pipeline_profile_online:online;
    auto &pipe_pv=pipeline_profile?s.pipeline_profile_pv:selected_pv;
    if(pipeline){
        ++s.pipeline_calls;s.pipeline_blocks+=(cfg.s+cfg.block-1)/cfg.block;
        // The first KQ shares the preparation stream and joins it once.
        // Subsequent SDNN work owns the same L1 banks, in queue order.
        auto kq_launch=[&](int s0,void *slot,XPUStream stream){int count=std::min(cfg.block,cfg.s-s0);flash_kv_launch(pipe_kq,{native_arg(slot),native_arg(k),native_arg(q),native_arg(kp),native_arg(qr),native_arg(cp),native_arg(s0),native_arg(count)},kv_kg,stream,timeline,s0/cfg.block*3);};
        auto online_launch=[&](int s0,int count,void *slot,XPUStream stream,int event_slot){const void *mp=count&&p.has_mask?mask:nullptr;native_launch_stream(pipe_online,{native_arg(y),native_arg(slot),native_arg(acc),native_arg(stats),native_arg(partial),native_arg(slot),native_arg(mp),native_arg(qr),native_arg(cp),native_arg(s0),native_arg(count)},16,stream,timeline,event_slot);};
        auto pv_launch=[&](int s0,int count,void *slot,int event_slot){flash_kv_launch(pipe_pv,{native_arg(partial),native_arg(v),native_arg(slot),native_arg(vmax),native_arg(cp),native_arg(s0),native_arg(count)},kv_vg,s.pipeline_sd,timeline,event_slot);};
        kq_launch(0,score,nullptr);check(xpu_wait());
        for(int s0=0,index=0;s0<cfg.s;s0+=cfg.block,++index){
            int count=std::min(cfg.block,cfg.s-s0);void *slot=index%2?next_score:score;
            if(s0+cfg.block>=cfg.s){
                // Drain the last online/PV/final chain in a single stream join.
                online_launch(s0,count,slot,s.pipeline_sd,index*3+1);
                pv_launch(s0,count,slot,index*3+2);
                online_launch(cfg.s,0,slot,s.pipeline_sd,(index+1)*3);
                check(xpu_wait(s.pipeline_sd));
            }else{
                online_launch(s0,count,slot,s.pipeline_cl,index*3+1);
                kq_launch(s0+cfg.block,index%2?score:next_score,s.pipeline_sd);
                check(xpu_wait(s.pipeline_cl));
                // The SDNN queue orders next KQ before PV. The CL join makes P ready.
                pv_launch(s0,count,slot,index*3+2);
                check(xpu_wait(s.pipeline_sd));
            }
        }
        if(pipeline_profile){
            std::vector<unsigned int> data(s.pipeline_timeline->size()/4);
            check(xpu_memcpy(data.data(),timeline,s.pipeline_timeline->size(),XPU_DEVICE_TO_HOST));check(xpu_wait());
            for(int slot=0;slot<(cfg.s+cfg.block-1)/cfg.block*3+1;++slot)for(int worker=0;worker<64;++worker){
                bool cl=slot%3==1||slot==(cfg.s+cfg.block-1)/cfg.block*3;if(!cl&&worker%16)continue;
                auto *t=data.data()+slot*1024+worker*16;
                std::fprintf(stderr,"FLASH_TIMELINE slot=%d worker=%d begin=%u end=%u ticks=%u\n",slot,worker,t[0],t[1],t[1]-t[0]);
            }
        }
    }else{
        for(int s0=0;s0<cfg.s;s0+=cfg.block){int count=std::min(cfg.block,cfg.s-s0);
            flash_kv_launch(*kq_program,{native_arg(score),native_arg(k),native_arg(q),native_arg(kp),native_arg(qr),native_arg(cp),native_arg(s0),native_arg(count)},kv_kg);stage("kq");
            const void *mp=p.has_mask?mask:nullptr;
            native_launch(online,{native_arg(y),native_arg(prob),native_arg(acc),native_arg(stats),native_arg(partial),native_arg(score),native_arg(mp),native_arg(qr),native_arg(cp),native_arg(s0),native_arg(count)},16);stage("online");
            flash_kv_launch(selected_pv,{native_arg(partial),native_arg(v),native_arg(prob),native_arg(vmax),native_arg(cp),native_arg(s0),native_arg(count)},kv_vg);stage("pv");
        }
        int s0=cfg.s,count=0;const void *mp=nullptr;
        native_launch(online,{native_arg(y),native_arg(prob),native_arg(acc),native_arg(stats),native_arg(partial),native_arg(score),native_arg(mp),native_arg(qr),native_arg(cp),native_arg(s0),native_arg(count)},16);check(xpu_wait());stage("final");
    }
    if(!s.calls++||trace)std::fprintf(stderr,"KUNLUN Flash PD%d D=%d S=%d M=%d H=%d block=%d workspace=%zu storage=%s kq=%s pipeline=%d second_score=%s calls=%zu\n",id,cfg.d,cfg.s,cfg.m,cfg.h,cfg.block,total,hbm?"HBM":"L3",kq_optimized?"reuse1024":"legacy",pipeline,pipeline?(state_hbm?"state-HBM":(both_l3?"L3":"HBM")):"none",s.calls);
    return true;
}

FlashStatistics flash_statistics(const Kernel &kernel){
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!kernel.impl_->flash)return {};
    auto &s=*std::static_pointer_cast<FlashState>(kernel.impl_->flash);size_t bytes=0;
    for(auto *p:{&s.qpack,&s.kpack,&s.vpack,&s.range})bytes+=p->scratch_size+p->quant_size+p->alias_input_size;
    for(auto *p:{s.config.get(),s.peaks.get(),s.ranges.get(),s.hbm.get(),s.pipeline_score.get(),s.pipeline_timeline.get()})if(p)bytes+=p->size();
    return {s.calls,bytes,s.last_l3?s.workspace:0,s.last_block,s.last_qt};
}
