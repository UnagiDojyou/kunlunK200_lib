// Dedicated Q6_K implementation. Q4 kernels, ABI and activation leases are unchanged.
namespace {
bool q6_overlap(const void *a, size_t as, const void *b, size_t bs) {
    auto aa=(uintptr_t)a, bb=(uintptr_t)b;
    require(aa<=UINTPTR_MAX-as && bb<=UINTPTR_MAX-bs);
    return aa<bb+bs && bb<aa+as;
}
std::vector<unsigned char> q6_pack(const void *raw, size_t bytes, const Q6KInfo &a) {
    require(raw && bytes==(size_t)a.n*(a.k/256)*210);
    std::vector<unsigned char> pack(a.payload_bytes,0);
    const int groups=a.k/16;
    for (int row=0;row<a.n;++row) {
        for (int block=0;block<a.k/256;++block) {
            auto *b=(const unsigned char*)raw+((size_t)row*(a.k/256)+block)*210;
            unsigned h=b[208]|b[209]<<8;
            int exponent=(h>>10)&31;
            require(exponent!=31);
            float d=std::ldexp(float((h&1023)+(exponent?1024:0)),exponent?exponent-25:-24);
            if(h&32768)d=-d;
            for (int subgroup=0;subgroup<16;++subgroup) {
                int g=block*16+subgroup;
                // A product of an FP16 value and signed INT8 is exact in FP32.
                float scale=d*(int8_t)b[192+subgroup];
                size_t ix=((size_t)(row/32)*groups+g)*32+row%32;
                std::memcpy(pack.data()+a.scale_offset+ix*4,&scale,4);
                for (int j=0;j<16;++j) {
                    int p=subgroup*16+j, half=p/128, l=p%32, lane=p%128/32;
                    int lo=half*64+l+(lane%2)*32;
                    int q=((lane<2?b[lo]&15:b[lo]>>4)|(((b[128+half*32+l]>>(lane*2))&3)<<4))-32;
                    pack[((size_t)(row/32)*groups+g)*512+j*32+row%32]=(unsigned char)(int8_t)q;
                }
            }
        }
    }
    return pack;
}
}
bool q6_k_supported(int k, int n) {
    return k>=256 && k<=4096 && k%256==0 && n>=32 && n<=32000 && n%32==0;
}
size_t q6_k_size(int k, int n) {
    return q6_k_supported(k,n)?aligned((size_t)k*n)+(size_t)(k/16)*n*4:0;
}
struct Q6KWeight::Impl {
    Q6KInfo info;
    Q6KStatistics stats{};
    std::unique_ptr<Buffer> owned;
    void *storage;
    bool ready=false;
};
Q6KWeight::Q6KWeight(int id, int k, int n, void *storage, size_t bytes): impl_(new Impl{}) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    require(id>=0 && id<2 && q6_k_supported(k,n));
    auto &s=*impl_;
    s.info={1,id,k,n,aligned((size_t)k*n),q6_k_size(k,n)};
    if (!storage) {
        s.owned.reset(new Buffer(id,s.info.payload_bytes));
        storage=s.owned->data();bytes=s.info.payload_bytes;
    }
    require((uintptr_t)storage%256==0 && bytes>=s.info.payload_bytes);
    s.storage=storage;
}
Q6KWeight::~Q6KWeight() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    select(impl_->info.device);check(xpu_wait());delete impl_;
}
Q6KInfo Q6KWeight::info() const { return impl_->info; }
void *Q6KWeight::data() const { return impl_->storage; }
Q6KStatistics Q6KWeight::statistics() const {
    std::lock_guard<std::recursive_mutex> lock(mutex);return impl_->stats;
}
void Q6KWeight::upload(const void *raw, size_t bytes) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto &s=*impl_;s.ready=false;
    auto begin=std::chrono::steady_clock::now();
    auto pack=q6_pack(raw,bytes,s.info);
    s.stats.pack_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    select(s.info.device);check(xpu_wait());begin=std::chrono::steady_clock::now();
    check(xpu_memcpy(s.storage,pack.data(),pack.size(),XPU_HOST_TO_DEVICE));check(xpu_wait());
    s.stats.upload_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    ++s.stats.uploads;s.stats.upload_bytes+=pack.size();s.ready=true;
}
struct Q6KWorkspace::Impl {
    int id=-1;
    Kernel::Impl programs[4];
    std::unique_ptr<Buffer> quant,scales,partial;
    Q6KStatistics stats{};
    size_t legacy_calls=0;
};
Q6KWorkspace::Q6KWorkspace(): impl_(new Impl{}) {}
Q6KWorkspace::~Q6KWorkspace() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (impl_->id>=0) {
        select(impl_->id);check(xpu_wait());
        for(auto &p:impl_->programs)if(p.code)check(xpu_free(p.code));
    }
    delete impl_;
}
Q6KStatistics Q6KWorkspace::statistics() const {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto stats=impl_->stats;
    for(auto *b:{impl_->quant.get(),impl_->scales.get(),impl_->partial.get()})if(b)stats.workspace_bytes+=b->size();
    return stats;
}
size_t Q6KWorkspace::legacy_calls() const {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return impl_->legacy_calls;
}
void q6_k_matmul(Q6KWorkspace &workspace, int id, void *y, const Q6KWeight &weight,
                 const void *x, int m, InputType type) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto &s=*workspace.impl_;auto a=weight.info();
    require(a.abi_version==1 && weight.impl_->ready && a.device==id && q6_k_supported(a.k,a.n));
    require(m>0 && m<=512 && (type==InputType::f32 || type==InputType::f16));
    int dtype=type==InputType::f16;
    require(id>=0 && id<2 && y && x && (uintptr_t)y%64==0 && (uintptr_t)x%(dtype?2:4)==0);
    size_t xb=(size_t)a.k*m*(dtype?2:4),yb=(size_t)a.n*m*4;
    require(!q6_overlap(x,xb,y,yb) && !q6_overlap(x,xb,weight.data(),a.payload_bytes) && !q6_overlap(y,yb,weight.data(),a.payload_bytes));
    if(s.id<0)s.id=id;
    require(s.id==id);select(id);
    native_reserve(s.quant,id,(size_t)(a.k/16)*1024);
    native_reserve(s.scales,id,(a.k/16)*4);
    const char *legacy_flag=std::getenv("KUNLUN_Q6_K_LEGACY");
    bool legacy=legacy_flag && std::strcmp(legacy_flag,"1")==0;
    if(legacy)native_reserve(s.partial,id,(size_t)(a.k/16)*a.n*4);
    else s.partial.reset();
    load_kernel(s.programs[0],"kunlun-q6-quant",id,false);
    if(legacy){load_kernel(s.programs[1],"kunlun-q6-sdnn",id,true);load_kernel(s.programs[2],"kunlun-q6-join",id,false);}
    else load_kernel(s.programs[3],"kunlun-q6-onchip",id,true);
    void *q=s.quant->data(),*scales=s.scales->data();
    const void *w=weight.data(),*ws=(const char*)w+a.scale_offset;
    for (int row=0;row<m;++row) {
        const void *input=(const char*)x+(size_t)row*a.k*(dtype?2:4);
        void *output=(char*)y+(size_t)row*a.n*4;
        native_launch(s.programs[0],{native_arg(q),native_arg(scales),native_arg(input),native_arg(a.k),native_arg(dtype)},16);
        void *target=legacy?s.partial->data():output;
        native_launch(s.programs[legacy?1:3],{native_arg(target),native_arg(w),native_arg(q),native_arg(a.k),native_arg(a.n),native_arg(ws),native_arg(scales)},8);
        if(legacy)native_launch(s.programs[2],{native_arg(output),native_arg(target),native_arg(a.k),native_arg(a.n)},16);
        // Default-stream order completes every consumer before buffer reuse/free.
        check(xpu_wait());++s.stats.calls;if(legacy)++s.legacy_calls;
    }
}
