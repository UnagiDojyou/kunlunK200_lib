#include "../kernels/attention-config.h"

namespace {
static_assert(sizeof(KunlunAttentionConfig)==160,"Attention launch ABI");
int tensor4_bytes(const Tensor4Layout &p) { return p.type==InputType::f16?2:4; }
size_t tensor4_span(const Tensor4Layout &p) {
    size_t size=tensor4_bytes(p);for(int d=0;d<4;++d)size+=(size_t)(p.ne[d]-1)*p.nb[d];return size;
}
size_t tensor4_count(const Tensor4Layout &p) {
    size_t n=1;for(int d=0;d<4;++d)n*=p.ne[d];return n;
}
bool tensor4_pointer(const void *ptr,size_t bytes,const Tensor4Layout &p) {
    uintptr_t a=(uintptr_t)ptr;size_t n=tensor4_span(p);
    return ptr && a%tensor4_bytes(p)==0 && n<=bytes && a+n>=a;
}
bool tensor4_overlap(const void *a,const Tensor4Layout &pa,const void *b,const Tensor4Layout &pb) {
    uintptr_t ap=(uintptr_t)a,bp=(uintptr_t)b;return ap<bp+tensor4_span(pb) && bp<ap+tensor4_span(pa);
}
bool tensor4_same(const Tensor4Layout &a,const Tensor4Layout &b) {
    return a.type==b.type && std::equal(a.ne,a.ne+4,b.ne) && std::equal(a.nb,a.nb+4,b.nb);
}
KunlunAttentionConfig attention_config(const Tensor4Layout &y,const Tensor4Layout &a) {
    KunlunAttentionConfig p{};std::copy(y.ne,y.ne+4,p.dims);std::copy(a.nb,a.nb+4,p.a);std::copy(y.nb,y.nb+4,p.y);
    p.type_a=a.type==InputType::f16?1:0;return p;
}
Kernel::Impl & attention_program(Kernel::Impl &parent,Kernel::Impl *Kernel::Impl::*member,int id) {
    if(!(parent.*member))parent.*member=new Kernel::Impl{};
    parent.id=id;
    (parent.*member)->id=id;
    return *(parent.*member);
}
void attention_launch(Kernel::Impl &state,int id,const char *name,const KunlunAttentionConfig &p,
                      std::initializer_list<const void*> pointers) {
    select(id);load_kernel(state,name,id);size_t off=0;
    for(const void *ptr:pointers){check(xpu_launch_argument_set(&ptr,8,off));off+=8;}
    if(state.attention_config_cache.size()!=sizeof(p) || std::memcmp(state.attention_config_cache.data(),&p,sizeof(p))) {
        if(!state.alias_input){state.alias_input=allocate(id,256);state.alias_input_size=256;require(state.alias_input);}
        check(xpu_memcpy(state.alias_input,&p,sizeof(p),XPU_HOST_TO_DEVICE));check(xpu_wait());
        state.attention_config_cache.assign((const unsigned char*)&p,(const unsigned char*)&p+sizeof(p));
    }
    check(xpu_launch_argument_set(&state.alias_input,8,off));check(xpu_launch_config(4,16));check(xpu_launch_async(state.fn));
}
Tensor4Layout attention_dense(const Tensor4Layout &input) {
    auto p=input;p.type=InputType::f32;int stride=4;
    for(int d=0;d<4;++d){p.nb[d]=stride;stride*=p.ne[d];}return p;
}
const void *attention_pack(Kernel::Impl &parent,Kernel::Impl *Kernel::Impl::*member,int id,const void *x,Tensor4Layout &layout,bool round) {
    auto dense=attention_dense(layout);size_t size=tensor4_count(dense)*4;
    auto &state=attention_program(parent,member,id);
    if(state.scratch_size<size){release(id,state.scratch,state.scratch_size);state.scratch=allocate(id,size);state.scratch_size=size;require(state.scratch);}
    const char *mode=std::getenv("KUNLUN_ATTENTION_PACK");
    if(mode && !std::strcmp(mode,"cl")) {
        auto cfg=attention_config(dense,layout);cfg.round_b=round;
        attention_launch(state,id,"kunlun-attention-pack",cfg,{state.scratch,x});
    } else {
        size_t count=tensor4_count(layout),half_bytes=count*2;
        if(!state.vendor){select(id);state.vendor=xdnn::create_context();require(state.vendor);state.vendor->set_ncluster(4);}
        bool contiguous=true;int stride=tensor4_bytes(layout);
        for(int d=0;d<4;++d){if(layout.ne[d]>1 && layout.nb[d]!=stride)contiguous=false;stride*=layout.ne[d];}
        auto half_buffer=[&]{if(state.quant_size<half_bytes){release(id,state.quant,state.quant_size);state.quant=allocate(id,half_bytes);state.quant_size=half_bytes;require(state.quant);}};
        const void *source=x;
        if(!contiguous) {
            auto flat=layout;int step=tensor4_bytes(layout);for(int d=0;d<4;++d){flat.nb[d]=step;step*=flat.ne[d];}
            auto cfg=attention_config(flat,layout);auto &gather=attention_program(parent,&Kernel::Impl::attention_cp,id);
            void *target=state.scratch;if(layout.type==InputType::f16){half_buffer();target=state.quant;}
            attention_launch(gather,id,"kunlun-attention-copy",cfg,{target,x});check(xpu_wait());source=target;
        }
        select(id);
        if(layout.type==InputType::f16) {
            check(xdnn::cast<float16,float>(state.vendor,(const float16*)source,(float*)state.scratch,(int)count));
        } else {
            half_buffer();
            check(xdnn::cast<float,float16>(state.vendor,(const float*)source,(float16*)state.quant,(int)count));
            check(xdnn::cast<float16,float>(state.vendor,(const float16*)state.quant,(float*)state.scratch,(int)count));
        }
        check(xpu_wait());
    }
    layout=dense;++parent.attention_pack_calls;return state.scratch;
}
}
const char * tensor4_reason(const Tensor4Layout &p) {
    if(p.type!=InputType::f32 && p.type!=InputType::f16)return "type";
    int bytes=tensor4_bytes(p);size_t count=1;
    for(int d=0;d<4;++d) {
        if(p.ne[d]<=0 || p.ne[d]>65536 || count>(size_t)INT_MAX/bytes/p.ne[d])return "extent";
        count*=p.ne[d];if(p.nb[d]<=0 || p.nb[d]%bytes)return "stride alignment";
    }
    if(p.nb[0]!=bytes || tensor4_span(p)>INT_MAX)return "inner stride or byte extent";
    int order[4]={0,1,2,3};std::sort(order,order+4,[&](int a,int b){return p.nb[a]<p.nb[b];});size_t span=bytes;
    for(int d:order)if(p.ne[d]>1){if((size_t)p.nb[d]<span)return "overlapping rows";span+=(size_t)(p.ne[d]-1)*p.nb[d];}
    return "";
}
const char * attention_matmul_reason(const Matmul4Layout &p) {
    for(const auto *v:{&p.a,&p.b,&p.y})if(*tensor4_reason(*v))return tensor4_reason(*v);
    if(p.y.type!=InputType::f32)return "output type";
    if(p.a.ne[0]!=p.b.ne[0] || p.a.ne[1]!=p.y.ne[0] || p.b.ne[1]!=p.y.ne[1] || p.b.ne[2]!=p.y.ne[2] || p.b.ne[3]!=p.y.ne[3])return "matrix shape";
    if(p.b.ne[2]%p.a.ne[2] || p.b.ne[3]%p.a.ne[3])return "GQA broadcast";
    if(p.a.ne[0]>16384 || p.a.ne[1]>16384 || p.b.ne[1]>256 || p.b.ne[2]>128 || p.b.ne[3]>32)return "validated matrix bounds";
    if(p.a.type==InputType::f16 && tensor4_count(p.a)>size_t(64)*1024*1024/4)return "A workspace exceeds 64 MiB";
    if((p.b.type==InputType::f16 || (p.a.type==InputType::f16 && p.round_b_to_f16)) && tensor4_count(p.b)>size_t(64)*1024*1024/4)return "B workspace exceeds 64 MiB";
    return "";
}
bool attention_matmul(Kernel &kernel,int id,void *y,size_t yb,const void *a,size_t ab,const void *b,size_t bb,const Matmul4Layout &p) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(*attention_matmul_reason(p) || id<0 || id>1 || (kernel.impl_->id!=-1 && kernel.impl_->id!=id))return false;
    if(!tensor4_pointer(y,yb,p.y) || !tensor4_pointer(a,ab,p.a) || !tensor4_pointer(b,bb,p.b) || tensor4_overlap(y,p.y,a,p.a) || tensor4_overlap(y,p.y,b,p.b))return false;
    auto pa=p.a,pb=p.b;
    if(pa.type==InputType::f16)a=attention_pack(*kernel.impl_,&Kernel::Impl::attention_pack_a,id,a,pa,false);
    if(pb.type==InputType::f16 || (p.a.type==InputType::f16 && p.round_b_to_f16))b=attention_pack(*kernel.impl_,&Kernel::Impl::attention_pack_b,id,b,pb,p.a.type==InputType::f16 && p.round_b_to_f16);
    auto cfg=attention_config(p.y,pa);std::copy(pb.nb,pb.nb+4,cfg.b);
    cfg.k=p.a.ne[0];cfg.n=p.a.ne[1];cfg.m=p.b.ne[1];cfg.heads_a=p.a.ne[2];cfg.batches_a=p.a.ne[3];cfg.round_b=0;
    auto &state=attention_program(*kernel.impl_,&Kernel::Impl::attention_mm,id);
    attention_launch(state,id,"kunlun-attention-matmul",cfg,{y,a,b});check(xpu_wait());++kernel.impl_->attention_calls[0];return true;
}
const char * attention_softmax_reason(const Softmax4Layout &p) {
    for(const auto *v:{&p.x,&p.y})if(*tensor4_reason(*v))return tensor4_reason(*v);
    if(p.x.type!=InputType::f32 || p.y.type!=InputType::f32 || !std::equal(p.x.ne,p.x.ne+4,p.y.ne))return "softmax shape/type";
    if(p.x.ne[0]>16384 || !std::isfinite(p.scale) || !std::isfinite(p.max_bias) || p.max_bias<0)return "softmax parameters";
    if(p.has_mask) {
        if(*tensor4_reason(p.mask))return tensor4_reason(p.mask);
        if(p.mask.ne[0]!=p.x.ne[0] || p.mask.ne[1]<p.x.ne[1])return "mask shape";
    }
    return "";
}
bool attention_softmax(Kernel &kernel,int id,void *y,size_t yb,const void *x,size_t xb,const void *mask,size_t mb,const Softmax4Layout &p) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(*attention_softmax_reason(p) || id<0 || id>1 || (kernel.impl_->id!=-1 && kernel.impl_->id!=id))return false;
    if(!tensor4_pointer(y,yb,p.y) || !tensor4_pointer(x,xb,p.x) || (tensor4_overlap(y,p.y,x,p.x) && !(y==x && tensor4_same(p.x,p.y))))return false;
    if(p.has_mask && (!tensor4_pointer(mask,mb,p.mask) || tensor4_overlap(y,p.y,mask,p.mask)))return false;
    auto cfg=attention_config(p.y,p.x);cfg.scale=p.scale;cfg.max_bias=p.max_bias;
    for(int d=0;d<4;++d){cfg.mask[d]=p.has_mask?p.mask.nb[d]:0;cfg.mask_ne[d]=p.has_mask?p.mask.ne[d]:1;}
    cfg.type_b=p.has_mask && p.mask.type==InputType::f16?1:0;
    auto &state=attention_program(*kernel.impl_,&Kernel::Impl::attention_sm,id);
    attention_launch(state,id,"kunlun-attention-softmax",cfg,{y,x,p.has_mask?mask:nullptr});check(xpu_wait());++kernel.impl_->attention_calls[1];return true;
}
bool attention_contiguous(Kernel &kernel,int id,void *y,size_t yb,const void *x,size_t xb,const Tensor4Layout &p) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(*tensor4_reason(p) || id<0 || id>1 || (kernel.impl_->id!=-1 && kernel.impl_->id!=id))return false;
    auto flat=p;int stride=tensor4_bytes(p);for(int d=0;d<4;++d){flat.nb[d]=stride;stride*=p.ne[d];}
    if(!tensor4_pointer(x,xb,p) || !tensor4_pointer(y,yb,flat))return false;
    if(tensor4_overlap(y,flat,x,p)){if(y==x && tensor4_same(p,flat))return true;return false;}
    auto cfg=attention_config(flat,p);auto &state=attention_program(*kernel.impl_,&Kernel::Impl::attention_cp,id);
    attention_launch(state,id,"kunlun-attention-copy",cfg,{y,x});check(xpu_wait());++kernel.impl_->attention_calls[2];return true;
}
const char * attention_rope_reason(const Rope4Layout &p) {
    for(const auto *v:{&p.x,&p.y})if(*tensor4_reason(*v))return tensor4_reason(*v);
    if(p.x.type!=p.y.type || !std::equal(p.x.ne,p.x.ne+4,p.y.ne) || p.x.ne[3]!=1 || p.x.ne[0]>512)return "rope shape/type";
    if(p.n_dims<=0 || p.n_dims%2 || p.n_offs<0 || p.n_offs%2 || p.n_dims>p.x.ne[0]-p.n_offs || (p.mode!=0 && p.mode!=2))return "rope rotation";
    if(!std::isfinite(p.freq_base) || p.freq_base<=1 || !std::isfinite(p.freq_scale) || p.freq_scale<=0 || !std::isfinite(p.attn_factor) || p.attn_factor<=0)return "rope parameters";
    return "";
}
bool attention_rope(Kernel &kernel,int id,void *y,size_t yb,const void *x,size_t xb,const int32_t *positions,size_t pb,const Rope4Layout &p) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(*attention_rope_reason(p) || id<0 || id>1 || (kernel.impl_->id!=-1 && kernel.impl_->id!=id))return false;
    if(!tensor4_pointer(y,yb,p.y) || !tensor4_pointer(x,xb,p.x) || !positions || (uintptr_t)positions%4 || pb<(size_t)p.x.ne[2]*4 || (uintptr_t)positions+pb<(uintptr_t)positions)return false;
    if(tensor4_overlap(y,p.y,x,p.x) && !(y==x && tensor4_same(p.x,p.y)))return false;
    uintptr_t yp=(uintptr_t)y,pp=(uintptr_t)positions;if(yp<pp+pb && pp<yp+tensor4_span(p.y))return false;
    // Bound range reduction before submitting either RoPE kernel. Only position IDs are read back.
    std::vector<int32_t> pos(p.x.ne[2]);select(id);check(xpu_memcpy(pos.data(),positions,pos.size()*4,XPU_DEVICE_TO_HOST));check(xpu_wait());
    bool host_trig=false;for(int32_t value:pos)host_trig=host_trig || std::abs(double(value)*p.freq_scale)>32768;
    auto cfg=attention_config(p.y,p.x);cfg.n_dims=p.n_dims;cfg.n_offs=p.n_offs;cfg.mode=p.mode;
    cfg.theta_scale=std::pow(p.freq_base,-2.0f/p.n_dims);cfg.freq_scale=p.freq_scale;cfg.attn_factor=p.attn_factor;
    auto &phase=attention_program(*kernel.impl_,&Kernel::Impl::attention_phase,id);
    auto &rotate=attention_program(*kernel.impl_,&Kernel::Impl::attention_rope,id);
    size_t size=(size_t)p.n_dims*p.x.ne[2]*4;
    if(phase.scratch_size<size){release(id,phase.scratch,phase.scratch_size);phase.scratch=allocate(id,size);phase.scratch_size=size;require(phase.scratch);}
    if(host_trig) {
        std::vector<float> trig((size_t)p.n_dims*p.x.ne[2]);
        for(int t=0;t<p.x.ne[2];++t){float theta=(float)pos[t];for(int j=0;j<p.n_dims/2;++j){float angle=theta*p.freq_scale;trig[(size_t)t*p.n_dims+2*j]=std::cos(angle)*p.attn_factor;trig[(size_t)t*p.n_dims+2*j+1]=std::sin(angle)*p.attn_factor;theta*=cfg.theta_scale;}}
        select(id);check(xpu_memcpy(phase.scratch,trig.data(),size,XPU_HOST_TO_DEVICE));check(xpu_wait());++kernel.impl_->attention_host_trig;
    } else attention_launch(phase,id,"kunlun-attention-rope-phase",cfg,{phase.scratch,positions});
    attention_launch(rotate,id,"kunlun-attention-rope",cfg,{y,x,phase.scratch});check(xpu_wait());++kernel.impl_->attention_calls[3];return true;
}
AttentionStatistics attention_statistics(const Kernel &kernel) {
    std::lock_guard<std::recursive_mutex> lock(mutex);auto *p=kernel.impl_;
    size_t bytes=p->attention_phase?p->attention_phase->scratch_size:0;
    for(auto *state:{p->attention_pack_a,p->attention_pack_b,p->attention_mm,p->attention_sm,p->attention_cp,p->attention_phase,p->attention_rope})if(state)bytes+=state->alias_input_size+state->quant_size+((state==p->attention_pack_a || state==p->attention_pack_b)?state->scratch_size:0);
    return {p->attention_calls[0],p->attention_calls[1],p->attention_calls[2],p->attention_calls[3],bytes,0,p->attention_host_trig,p->attention_pack_calls};
}
