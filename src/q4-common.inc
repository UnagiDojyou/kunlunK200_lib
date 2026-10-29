// Private implementation shares the existing Program, allocation and L3 runtime.
#include "q4-common-pack.h"

bool common_q4_supported(int k, int n) { return k > 0 && k <= 16384 && n > 0 && n <= 32000; }
size_t common_q4_size(int k, int n) {
    if (!common_q4_supported(k,n)) return 0;
    return common_q4::describe(k,n,common_q4::Layout::Micro32Bank,common_q4::ScaleOrder::KMajor).payload_bytes;
}
struct CommonQ4Weight::Impl {
    CommonQ4Info info;
    std::unique_ptr<Buffer> owned;
    void * storage;
    bool ready = false;
    CommonQ4WeightStatistics stats{};
};
CommonQ4Weight::CommonQ4Weight(int id, int k, int n, void * storage, size_t bytes): impl_(new Impl{}) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    require(id >= 0 && id < 2 && common_q4_supported(k,n));
    auto a=common_q4::describe(k,n,common_q4::Layout::Micro32Bank,common_q4::ScaleOrder::KMajor);
    impl_->info={1,2,1,1,k,n,a.kp,a.np,id,a.scale_offset,a.payload_bytes};
    if (!storage) { impl_->owned.reset(new Buffer(id,a.payload_bytes)); storage=impl_->owned->data(); bytes=a.payload_bytes; }
    require((uintptr_t)storage%256 == 0 && bytes >= a.payload_bytes);
    impl_->storage=storage;
}
CommonQ4Weight::~CommonQ4Weight() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    select(impl_->info.device); check(xpu_wait()); delete impl_;
}
CommonQ4Info CommonQ4Weight::info() const { return impl_->info; }
void * CommonQ4Weight::data() const { return impl_->storage; }
CommonQ4WeightStatistics CommonQ4Weight::statistics() const { std::lock_guard<std::recursive_mutex> lock(mutex); return impl_->stats; }
void CommonQ4Weight::upload(const void * raw, size_t bytes) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto & s=*impl_; auto a=s.info;
    require(raw && bytes == (size_t)a.n*((a.k+31)/32)*18);
    auto begin=std::chrono::steady_clock::now();
    common_q4::Pack packed((const unsigned char *)raw,bytes,a.k,a.n,common_q4::Layout::Micro32Bank,common_q4::ScaleOrder::KMajor,false);
    s.stats.pack_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    select(a.device); check(xpu_wait()); begin=std::chrono::steady_clock::now();
    check(xpu_memcpy(s.storage,packed.bytes.data(),a.payload_bytes,XPU_HOST_TO_DEVICE)); check(xpu_wait());
    s.stats.upload_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    ++s.stats.uploads; s.stats.upload_bytes+=a.payload_bytes; s.ready=true;
}

namespace {
struct CommonActivationStorage {
    std::unique_ptr<Buffer> quant, scale;
    uint64_t epoch=0;
    bool valid=false, profile=false;
    CommonQ4ActivationStatistics stats{};
    CommonActivationStorage() {
        const char * p=std::getenv("KUNLUN_Q4_ACT_PROFILE"); profile=p && !std::strcmp(p,"1");
    }
};
void common_invalidate(CommonActivationStorage & s) {
    if (s.valid) ++s.stats.invalidations;
    s.valid=false; require(++s.epoch!=0);
}
uint64_t common_tick(bool enabled) {
    return enabled ? std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count() : 0;
}
}
struct CommonQ4ActivationLease::Impl {
    std::shared_ptr<CommonActivationStorage> storage;
    uint64_t epoch=0, generation=0;
    int id=-1, k=0, m=0;
    InputType type=InputType::f32;
    const void * input=nullptr;
    size_t input_bytes=0;
    std::vector<CommonQ4ActivationChunk> chunks;
};

struct CommonQ4Workspace::Impl {
    int id=-1;
    Kernel::Impl programs[19];
    std::unique_ptr<Buffer> input, output, peaks, rest, peak1, slice, ticks;
    std::shared_ptr<CommonActivationStorage> activation=std::make_shared<CommonActivationStorage>();
    const void * last_values=nullptr, * last_scales=nullptr;
    std::shared_ptr<NativeL3> l3;
    bool l3_unavailable=false;
    CommonQ4Statistics stats{};
    CommonQ4MacStatistics mac_stats{};
    bool mac_invalid_logged=false, mac_shape_logged=false, nmajor_invalid_logged=false, nmajor_shape_logged=false;
    int last_rows=0, last_kp=0, last_components=0;
};
CommonQ4Workspace::CommonQ4Workspace(): impl_(new Impl{}) {}
CommonQ4Workspace::~CommonQ4Workspace() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    common_invalidate(*impl_->activation);
    if (impl_->id >= 0) {
        select(impl_->id); check(xpu_wait());
        for (auto & p : impl_->programs) if (p.code) check(xpu_free(p.code));
    }
    delete impl_;
}
CommonQ4Quantization CommonQ4Workspace::quantization() const {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto & s=*impl_; return {s.last_components ? s.last_values : nullptr,s.last_components ? s.last_scales : nullptr,s.last_rows,s.last_kp,s.last_components};
}
CommonQ4Statistics CommonQ4Workspace::statistics() const {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto & s=*impl_; auto a=s.stats; a.hbm_workspace_bytes=0;
    for (auto * b : {s.input.get(),s.output.get(),s.activation->quant.get(),s.activation->scale.get(),s.peaks.get(),s.rest.get(),s.peak1.get(),s.slice.get(),s.ticks.get()})
        if (b) a.hbm_workspace_bytes+=std::max<size_t>(b->size(),256);
    a.l3_workspace_bytes=s.l3 ? s.l3->bytes : 0; a.slice_bytes=s.slice ? s.slice->size() : 0; return a;
}
CommonQ4MacStatistics CommonQ4Workspace::mac_statistics() const {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return impl_->mac_stats;
}
namespace {
bool common_stream_shape(const CommonQ4Info & a) {
    if(a.k!=a.kp)return false;
    static const int shapes[][2]={
        {2048,256},{2048,1024},{2048,2048},{2048,2816},{2048,5632},
        {4096,256},{4096,2048},{4096,4096},{4096,5632},{4096,11008},
        {5632,1024},{5632,2048},{11008,2048},{11008,4096}
    };
    for(const auto & shape:shapes)if(a.kp==shape[0] && a.np==shape[1])return true;
    return false;
}
bool common_stream_enabled(CommonQ4Workspace::Impl & s,const CommonQ4Info & a) {
    const char * mode=std::getenv("KUNLUN_Q4_MAC_DMA");
    if(!mode || !std::strcmp(mode,"0"))return false;
    if(std::strcmp(mode,"1")) {
        if(!s.mac_invalid_logged)std::fprintf(stderr,"Kunlun Q4 MAC/DMA: invalid KUNLUN_Q4_MAC_DMA=%s; using stable path\n",mode);
        s.mac_invalid_logged=true;return false;
    }
    if(common_stream_shape(a))return true;
    ++s.mac_stats.shape_fallbacks;
    if(!s.mac_shape_logged)std::fprintf(stderr,"Kunlun Q4 MAC/DMA PD%d: unvalidated shape K=%d N=%d Kp=%d Np=%d; using stable GEMV\n",a.device,a.k,a.n,a.kp,a.np);
    s.mac_shape_logged=true;return false;
}
bool common_nmajor_enabled(CommonQ4Workspace::Impl & s,const CommonQ4Info & a) {
    const char * mode=std::getenv("KUNLUN_Q4_NMAJOR");
    if(!mode || !std::strcmp(mode,"0"))return false;
    if(!std::strcmp(mode,"1")) {
        if((a.kp==4096 && (a.np==4096 || a.np==11008)) || (a.kp==11008 && a.np==4096))return true;
        if(!s.nmajor_shape_logged)std::fprintf(stderr,"Kunlun Q4 N-major PD%d: unvalidated shape K=%d Np=%d; using 0.7.8 stream order\n",s.id,a.kp,a.np);
        s.nmajor_shape_logged=true;return false;
    }
    if(!s.nmajor_invalid_logged)std::fprintf(stderr,"Kunlun Q4 N-major: invalid setting; using 0.7.8 stream order\n");
    s.nmajor_invalid_logged=true;return false;
}
Kernel::Impl & common_program(CommonQ4Workspace::Impl & s, int index, const char * name, bool sd=false) {
    auto & p=s.programs[index]; load_kernel(p,name,s.id,sd); return p;
}
void common_rows(CommonQ4Workspace::Impl & s, void * dst, const void * src, int width, int rows, int src_stride, int dst_stride, int dst_rows, int bytes) {
    auto & p=common_program(s,15,"kunlun-common-io");
    native_launch(p,{native_arg(dst),native_arg(src),native_arg(width),native_arg(rows),native_arg(src_stride),native_arg(dst_stride),native_arg(dst_rows),native_arg(bytes)},16);
    check(xpu_wait());
}
void common_quant_gemv(CommonQ4Workspace::Impl & s, const void * x, int k, int dtype, void * q, void * params) {
    uint64_t begin=common_tick(s.activation->profile);
    native_reserve(s.rest,s.id,(size_t)k*4);native_reserve(s.peaks,s.id,2048);native_reserve(s.peak1,s.id,2048);
    void * r=s.rest->data(), * p0=s.peaks->data(), * p1=s.peak1->data();
    native_launch(common_program(s,0,"kunlun-common-max"),{native_arg(p0),native_arg(x),native_arg(k),native_arg(dtype)},16);
    native_launch(common_program(s,1,"kunlun-common-main"),{native_arg(q),native_arg(r),native_arg(p1),native_arg(x),native_arg(p0),native_arg(k),native_arg(dtype)},16);
    native_launch(common_program(s,2,"kunlun-common-residual"),{native_arg(q),native_arg(params),native_arg(r),native_arg(p0),native_arg(p1),native_arg(k)},16);
    check(xpu_wait());
    ++s.activation->stats.quantized_chunks; s.activation->stats.quant_kernel_launches+=3;
    if (s.activation->profile) s.activation->stats.quantization_ns+=common_tick(true)-begin;
}
void common_quant_gemm(CommonQ4Workspace::Impl & s, const void * x, int m, int k, int dtype, void * q, void * scale) {
    uint64_t begin=common_tick(s.activation->profile);
    if (m >= 64) native_launch(common_program(s,7,"kunlun-common-quant-row-serial-fused"),{native_arg(q),native_arg(scale),native_arg(x),native_arg(m),native_arg(k),native_arg(dtype)},16);
    else {
        native_reserve(s.peaks,s.id,(size_t)m*((k/64+15)/16*16)*4);
        void * peaks=s.peaks->data();int mode=1;
        native_launch(common_program(s,5,"kunlun-common-quant-partial"),{native_arg(peaks),native_arg(x),native_arg(m),native_arg(k),native_arg(dtype)},16);
        native_launch(common_program(s,6,"kunlun-common-quant-row-fused"),{native_arg(q),native_arg(scale),native_arg(x),native_arg(peaks),native_arg(m),native_arg(k),native_arg(dtype),native_arg(mode)},16);
    }
    check(xpu_wait());
    ++s.activation->stats.quantized_chunks; s.activation->stats.quant_kernel_launches+=m>=64 ? 1 : 2;
    if (s.activation->profile) s.activation->stats.quantization_ns+=common_tick(true)-begin;
}
bool common_gemv(CommonQ4Workspace::Impl & s, void * y, const CommonQ4Info & a, const void * w, const void * x, int dtype, const CommonQ4ActivationChunk * prepared=nullptr) {
    int k=a.kp,n=a.np,batch=1,first=0,components=2,bits=8,pipelined=1,fold=0,mode=1;
    const void * scales=(const char *)w+a.scale_offset;
    size_t partial_bytes=(size_t)n*(k/32)*4;
    if (!s.l3) { s.l3=native_l3[s.id].lock(); if (!s.l3) { s.l3=std::make_shared<NativeL3>(s.id); native_l3[s.id]=s.l3; } }
    if (partial_bytes > 16*1024*1024 || s.l3_unavailable || !s.l3->reserve(partial_bytes)) {
        s.l3_unavailable=true;s.last_components=0;
        if (prepared) return false;
        auto & p=common_program(s,16,"kunlun-common-fallback");
        std::fprintf(stderr,"Kunlun common Q4 PD%d exact CL fallback: L3 capacity/allocation K=%d N=%d\n",s.id,k,n);
        native_launch(p,{native_arg(y),native_arg(w),native_arg(scales),native_arg(x),native_arg(k),native_arg(n),native_arg(batch),native_arg(dtype)},8);
        check(xpu_wait()); ++s.stats.paths[7]; return true;
    }
    if (!prepared) { native_reserve(s.activation->quant,s.id,(size_t)4*k+1024); native_reserve(s.activation->scale,s.id,64); }
    void * q=prepared ? const_cast<void *>(prepared->values) : s.activation->quant->data();
    void * params=prepared ? const_cast<void *>(prepared->scales) : s.activation->scale->data(), * partial=s.l3->pointer;
    if (!prepared) common_quant_gemv(s,x,k,dtype,q,params);
    s.last_values=q;s.last_scales=params;float zero=0,one=1;
    bool stream=common_stream_enabled(s,a);
    bool nmajor=stream && common_nmajor_enabled(s,a);
    if(nmajor && !s.programs[18].fn)std::fprintf(stderr,"Kunlun Q4 N-major PD%d: opt-in, ABI=1, unchanged partial order\n",s.id);
    if(stream && !nmajor && !s.programs[17].fn)std::fprintf(stderr,"Kunlun Q4 MAC/DMA stream PD%d: ABI=1 K=%d N=%d resident_L1D<=524288; original MAC order\n",s.id,a.k,a.n);
    native_launch(common_program(s,nmajor?18:stream?17:3,nmajor?"kunlun-common-gemv-nmajor":stream?"kunlun-common-gemv-stream":"kunlun-common-gemv",true),{native_arg(partial),native_arg(w),native_arg(q),native_arg(k),native_arg(n),native_arg(batch),native_arg(first),native_arg(n),native_arg(components),native_arg(bits),native_arg(zero),native_arg(zero),native_arg(one),native_arg(pipelined),native_arg(scales),native_arg(fold),native_arg(params)},8);
    check(xpu_wait());
    native_launch(common_program(s,4,"kunlun-common-sum"),{native_arg(y),native_arg(partial),native_arg(scales),native_arg(k),native_arg(n),native_arg(batch),native_arg(first),native_arg(n),native_arg(mode)},16);
    check(xpu_wait());s.last_components=2; ++s.stats.paths[0];
    if(stream){++s.mac_stats.stream_sd_calls;s.mac_stats.resident_l1d_bytes=std::max(s.mac_stats.resident_l1d_bytes,(size_t)std::min(k/32,128)*4096);}
    else ++s.mac_stats.baseline_sd_calls;
    return true;
}
void common_gemm(CommonQ4Workspace::Impl & s, void * y, const CommonQ4Info & a, const void * w, const void * x, int m, int dtype, const CommonQ4ActivationChunk * prepared=nullptr) {
    const int k=a.kp,n=a.np;
    int path,nt,kt,parts=1;
    const char * name;
    if (m == 8) { path=1;name="kunlun-common-combined";nt=n%512?256:512;kt=n%512?128:64; }
    else if (m == 32) { path=2;name="kunlun-common-wide";nt=64;kt=256; }
    else if (m == 64) { path=k>8192?5:3;name=k>8192?"kunlun-common-combo2":"kunlun-common-rows64";nt=64;kt=128;parts=k>8192?2:1; }
    else { require(m == 128);path=k>4096?6:4;name=k>4096?"kunlun-common-combo4":"kunlun-common-rows128";nt=32;kt=128;parts=k>4096?4:1; }
    // K partitions and capacity are the same as the validated Rows/Combo schedules.
    int max_groups=((k/kt)+parts-1)/parts,mt=m <= 32 ? 32 : m;
    while (mt>32 && mt*max_groups*kt>524288) mt/=2;
    require(mt*max_groups*kt<=524288 && n%nt==0 && k%kt==0 && nt*kt<=32768);
    if (!prepared) { native_reserve(s.activation->quant,s.id,(size_t)128*k+1024);native_reserve(s.activation->scale,s.id,128*64); }
    native_reserve(s.ticks,s.id,8192);
    void * q=prepared ? const_cast<void *>(prepared->values) : s.activation->quant->data();
    void * scale=prepared ? const_cast<void *>(prepared->scales) : s.activation->scale->data(), * ticks=s.ticks->data();
    if (!prepared) common_quant_gemm(s,x,m,k,dtype,q,scale);
    s.last_values=q;s.last_scales=scale;
    void * target=y;
    if (parts>1) { native_reserve(s.slice,s.id,(size_t)parts*m*n*4); target=s.slice->data(); }
    const void * scales=(const char *)w+a.scale_offset;
    // One typed argument sequence for all SD families; ABI is validated at entry.
    native_launch(common_program(s,8+path-1,name,true),{native_arg(target),native_arg(q),native_arg(w),native_arg(scales),native_arg(scale),native_arg(m),native_arg(n),native_arg(k),native_arg(nt),native_arg(kt),native_arg(ticks)},8);
    check(xpu_wait());
    if (parts>1) {
        // The seventh stats argument is mandatory in the non-profile join too.
        native_launch(common_program(s,14,"kunlun-common-join-slices"),{native_arg(y),native_arg(target),native_arg(scale),native_arg(m),native_arg(n),native_arg(parts),native_arg(ticks)},16);
        check(xpu_wait());
    }
    ++s.stats.paths[path];
}
}

namespace {
bool common_execute(CommonQ4Workspace::Impl & s, int id, void * y, const CommonQ4Info & a, const CommonQ4Weight::Impl & w,
                    const void * input, int m, InputType type, const CommonQ4ActivationLease::Impl * prepared=nullptr) {
    int bytes=type==InputType::f16?2:4,dtype=bytes==2;
    size_t chunk_index=0;
    for (int first=0;first<m;) {
        int rows=std::min(128,m-first),padded=m==1?1:rows<=8?8:rows<=32?32:rows<=64?64:128;
        const void * x=(const char *)input+(size_t)first*a.k*bytes;
        void * out=(float *)y+(size_t)first*a.n;
        if (!prepared && (a.k!=a.kp || rows!=padded || (uintptr_t)x%64)) {
            native_reserve(s.input,id,(size_t)padded*a.kp*bytes);
            common_rows(s,s.input->data(),x,a.k,rows,a.k,a.kp,padded,bytes); x=s.input->data();
        }
        bool crop=a.n!=a.np || rows!=padded || (uintptr_t)out%64;
        if (crop) { native_reserve(s.output,id,(size_t)padded*a.np*4); out=s.output->data(); }
        const auto * chunk=prepared ? &prepared->chunks[chunk_index++] : nullptr;
        if (m == 1) { if (!common_gemv(s,out,a,w.storage,x,dtype,chunk)) return false; }
        else common_gemm(s,out,a,w.storage,x,padded,dtype,chunk);
        if (crop) common_rows(s,(float *)y+(size_t)first*a.n,out,a.n,rows,a.np,a.n,rows,4);
        s.last_rows=rows;s.last_kp=a.kp;if(m!=1)s.last_components=1;
        first+=rows;++s.stats.chunks;
    }
    check(xpu_wait());++s.stats.calls;++matmuls[id];return true;
}
}
void common_q4_matmul(CommonQ4Workspace & workspace, int id, void * y, const CommonQ4Weight & weight, const void * input, int m, InputType type) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto & w=*weight.impl_; auto a=w.info;auto & s=*workspace.impl_;
    require(w.ready && id == a.device && common_q4_supported(a.k,a.n));
    auto expected=common_q4::describe(a.k,a.n,common_q4::Layout::Micro32Bank,common_q4::ScaleOrder::KMajor);
    require(a.abi_version==1 && a.quant_type==2 && a.layout==1 && a.scale_order==1 && a.kp==expected.kp && a.np==expected.np && a.scale_offset==expected.scale_offset && a.payload_bytes==expected.payload_bytes);
    require(m>0 && (size_t)m*a.k<=INT_MAX/4 && (size_t)m*a.n<=INT_MAX/4 && input && y && (type==InputType::f32 || type==InputType::f16));
    int bytes=type==InputType::f16?2:4;
    uintptr_t yp=(uintptr_t)y,xp=(uintptr_t)input,wp=(uintptr_t)w.storage;
    size_t ys=(size_t)m*a.n*4,xs=(size_t)m*a.k*bytes,ws=a.payload_bytes;
    require(yp%4==0 && xp%bytes==0 && wp%256==0 && yp+ys>=yp && xp+xs>=xp && wp+ws>=wp);
    require((yp+ys<=xp || xp+xs<=yp) && (yp+ys<=wp || wp+ws<=yp) && (xp+xs<=wp || wp+ws<=xp));
    require(s.id == -1 || s.id == id);s.id=id;select(id);
    common_invalidate(*s.activation);s.last_components=0;
    require(common_execute(s,id,y,a,w,input,m,type));
}

#include "q4-activation.inc"
