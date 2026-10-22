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
struct CommonQ4Workspace::Impl {
    int id=-1;
    Kernel::Impl programs[17];
    std::unique_ptr<Buffer> input, output, quant, scale, peaks, rest, peak1, slice, ticks;
    std::shared_ptr<NativeL3> l3;
    bool l3_unavailable=false;
    CommonQ4Statistics stats{};
    int last_rows=0, last_kp=0, last_components=0;
};
CommonQ4Workspace::CommonQ4Workspace(): impl_(new Impl{}) {}
CommonQ4Workspace::~CommonQ4Workspace() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (impl_->id >= 0) {
        select(impl_->id); check(xpu_wait());
        for (auto & p : impl_->programs) if (p.code) check(xpu_free(p.code));
    }
    delete impl_;
}
CommonQ4Quantization CommonQ4Workspace::quantization() const {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto & s=*impl_; return {s.last_components && s.quant ? s.quant->data() : nullptr,s.last_components && s.scale ? s.scale->data() : nullptr,s.last_rows,s.last_kp,s.last_components};
}
CommonQ4Statistics CommonQ4Workspace::statistics() const {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto & s=*impl_; auto a=s.stats; a.hbm_workspace_bytes=0;
    for (auto * b : {s.input.get(),s.output.get(),s.quant.get(),s.scale.get(),s.peaks.get(),s.rest.get(),s.peak1.get(),s.slice.get(),s.ticks.get()})
        if (b) a.hbm_workspace_bytes+=std::max<size_t>(b->size(),256);
    a.l3_workspace_bytes=s.l3 ? s.l3->bytes : 0; a.slice_bytes=s.slice ? s.slice->size() : 0; return a;
}
namespace {
Kernel::Impl & common_program(CommonQ4Workspace::Impl & s, int index, const char * name, bool sd=false) {
    auto & p=s.programs[index]; load_kernel(p,name,s.id,sd); return p;
}
void common_rows(CommonQ4Workspace::Impl & s, void * dst, const void * src, int width, int rows, int src_stride, int dst_stride, int dst_rows, int bytes) {
    auto & p=common_program(s,15,"kunlun-common-io");
    native_launch(p,{native_arg(dst),native_arg(src),native_arg(width),native_arg(rows),native_arg(src_stride),native_arg(dst_stride),native_arg(dst_rows),native_arg(bytes)},16);
    check(xpu_wait());
}
void common_gemv(CommonQ4Workspace::Impl & s, void * y, const CommonQ4Info & a, const void * w, const void * x, int dtype) {
    int k=a.kp,n=a.np,batch=1,first=0,components=2,bits=8,pipelined=1,fold=0,mode=1;
    const void * scales=(const char *)w+a.scale_offset;
    size_t partial_bytes=(size_t)n*(k/32)*4;
    if (!s.l3) { s.l3=native_l3[s.id].lock(); if (!s.l3) { s.l3=std::make_shared<NativeL3>(s.id); native_l3[s.id]=s.l3; } }
    if (partial_bytes > 16*1024*1024 || s.l3_unavailable || !s.l3->reserve(partial_bytes)) {
        s.l3_unavailable=true;s.last_components=0;
        auto & p=common_program(s,16,"kunlun-common-fallback");
        std::fprintf(stderr,"Kunlun common Q4 PD%d exact CL fallback: L3 capacity/allocation K=%d N=%d\n",s.id,k,n);
        native_launch(p,{native_arg(y),native_arg(w),native_arg(scales),native_arg(x),native_arg(k),native_arg(n),native_arg(batch),native_arg(dtype)},8);
        check(xpu_wait()); ++s.stats.paths[7]; return;
    }
    native_reserve(s.quant,s.id,(size_t)4*k+1024); native_reserve(s.rest,s.id,(size_t)k*4);
    native_reserve(s.peaks,s.id,2048);native_reserve(s.peak1,s.id,2048);native_reserve(s.scale,s.id,64);
    void * q=s.quant->data(), * r=s.rest->data(), * p0=s.peaks->data(), * p1=s.peak1->data(), * params=s.scale->data(), * partial=s.l3->pointer;
    native_launch(common_program(s,0,"kunlun-common-max"),{native_arg(p0),native_arg(x),native_arg(k),native_arg(dtype)},16);
    native_launch(common_program(s,1,"kunlun-common-main"),{native_arg(q),native_arg(r),native_arg(p1),native_arg(x),native_arg(p0),native_arg(k),native_arg(dtype)},16);
    native_launch(common_program(s,2,"kunlun-common-residual"),{native_arg(q),native_arg(params),native_arg(r),native_arg(p0),native_arg(p1),native_arg(k)},16);
    check(xpu_wait()); float zero=0,one=1;
    native_launch(common_program(s,3,"kunlun-common-gemv",true),{native_arg(partial),native_arg(w),native_arg(q),native_arg(k),native_arg(n),native_arg(batch),native_arg(first),native_arg(n),native_arg(components),native_arg(bits),native_arg(zero),native_arg(zero),native_arg(one),native_arg(pipelined),native_arg(scales),native_arg(fold),native_arg(params)},8);
    check(xpu_wait());
    native_launch(common_program(s,4,"kunlun-common-sum"),{native_arg(y),native_arg(partial),native_arg(scales),native_arg(k),native_arg(n),native_arg(batch),native_arg(first),native_arg(n),native_arg(mode)},16);
    check(xpu_wait());s.last_components=2; ++s.stats.paths[0];
}
void common_gemm(CommonQ4Workspace::Impl & s, void * y, const CommonQ4Info & a, const void * w, const void * x, int m, int dtype) {
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
    native_reserve(s.quant,s.id,(size_t)128*k+1024);native_reserve(s.scale,s.id,128*64);native_reserve(s.ticks,s.id,8192);
    void * q=s.quant->data(), * scale=s.scale->data(), * ticks=s.ticks->data();
    if (m >= 64) native_launch(common_program(s,7,"kunlun-common-quant-row-serial-fused"),{native_arg(q),native_arg(scale),native_arg(x),native_arg(m),native_arg(k),native_arg(dtype)},16);
    else {
        native_reserve(s.peaks,s.id,(size_t)m*((k/64+15)/16*16)*4);
        void * peaks=s.peaks->data();int mode=1;
        native_launch(common_program(s,5,"kunlun-common-quant-partial"),{native_arg(peaks),native_arg(x),native_arg(m),native_arg(k),native_arg(dtype)},16);
        native_launch(common_program(s,6,"kunlun-common-quant-row-fused"),{native_arg(q),native_arg(scale),native_arg(x),native_arg(peaks),native_arg(m),native_arg(k),native_arg(dtype),native_arg(mode)},16);
    }
    check(xpu_wait());
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
void common_q4_matmul(CommonQ4Workspace & workspace, int id, void * y, const CommonQ4Weight & weight, const void * input, int m, InputType type) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto & w=*weight.impl_; auto a=w.info;auto & s=*workspace.impl_;
    require(w.ready && id == a.device && common_q4_supported(a.k,a.n));
    auto expected=common_q4::describe(a.k,a.n,common_q4::Layout::Micro32Bank,common_q4::ScaleOrder::KMajor);
    require(a.abi_version==1 && a.quant_type==2 && a.layout==1 && a.scale_order==1 && a.kp==expected.kp && a.np==expected.np && a.scale_offset==expected.scale_offset && a.payload_bytes==expected.payload_bytes);
    require(m>0 && (size_t)m*a.k<=INT_MAX/4 && (size_t)m*a.n<=INT_MAX/4 && input && y && (type==InputType::f32 || type==InputType::f16));
    int bytes=type==InputType::f16?2:4,dtype=bytes==2;
    uintptr_t yp=(uintptr_t)y,xp=(uintptr_t)input,wp=(uintptr_t)w.storage;
    size_t ys=(size_t)m*a.n*4,xs=(size_t)m*a.k*bytes,ws=a.payload_bytes;
    require(yp%4==0 && xp%bytes==0 && wp%256==0 && yp+ys>=yp && xp+xs>=xp && wp+ws>=wp);
    require((yp+ys<=xp || xp+xs<=yp) && (yp+ys<=wp || wp+ws<=yp) && (xp+xs<=wp || wp+ws<=xp));
    require(s.id == -1 || s.id == id);s.id=id;select(id);
    for (int first=0;first<m;) {
        int rows=std::min(128,m-first),padded=m==1?1:rows<=8?8:rows<=32?32:rows<=64?64:128;
        const void * x=(const char *)input+(size_t)first*a.k*bytes;
        void * out=(float *)y+(size_t)first*a.n;
        if (a.k!=a.kp || rows!=padded || (uintptr_t)x%64) {
            native_reserve(s.input,id,(size_t)padded*a.kp*bytes);
            common_rows(s,s.input->data(),x,a.k,rows,a.k,a.kp,padded,bytes); x=s.input->data();
        }
        bool crop=a.n!=a.np || rows!=padded || (uintptr_t)out%64;
        if (crop) { native_reserve(s.output,id,(size_t)padded*a.np*4); out=s.output->data(); }
        if (m == 1) common_gemv(s,out,a,w.storage,x,dtype);
        else common_gemm(s,out,a,w.storage,x,padded,dtype);
        if (crop) common_rows(s,(float *)y+(size_t)first*a.n,out,a.n,rows,a.np,a.n,rows,4);
        s.last_rows=rows;s.last_kp=a.kp;if(m!=1)s.last_components=1;
        first+=rows;++s.stats.chunks;
    }
    check(xpu_wait());++s.stats.calls;++matmuls[id];
}
