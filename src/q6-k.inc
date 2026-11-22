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
    std::unique_ptr<Buffer> quant,scales,partial,shard_output;
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
    for(auto *b:{impl_->quant.get(),impl_->scales.get(),impl_->partial.get(),impl_->shard_output.get()})if(b)stats.workspace_bytes+=b->size();
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


namespace {
int q6_pair_launch(Kernel::Impl & program, std::initializer_list<NativeArgument> args, int cores) {
    size_t offset = 0;
    for (const auto & arg : args) {
        const int status = xpu_launch_argument_set(arg.value, arg.size, offset);
        if (status) return status;
        offset += arg.size;
    }
    const int status = xpu_launch_config(4, cores);
    return status ? status : xpu_launch_async(program.fn);
}
}

int q6_k_matmul_pair(Q6KWorkspace & ws0, Q6KWorkspace & ws1, void * y,
        const Q6KWeight & w0, const Q6KWeight & w1, const void * x0, const void * x1,
        Q6KPairTiming * timing) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    constexpr int k = 4096;
    constexpr int n = 32000;
    constexpr int half = n/2;
    constexpr int dtype = 0;
    constexpr size_t bytes = half*sizeof(float);
    constexpr size_t xb = k*sizeof(float);
    const char * legacy = std::getenv("KUNLUN_Q6_K_LEGACY");
    if (&ws0 == &ws1 || !y || !x0 || !x1 || (uintptr_t)y%256 ||
            (uintptr_t)x0%4 || (uintptr_t)x1%4 || (legacy && std::strcmp(legacy,"1") == 0)) return 0;
    Q6KWorkspace::Impl * work[2] = {ws0.impl_, ws1.impl_};
    const Q6KWeight * weights[2] = {&w0, &w1};
    const void * inputs[2] = {x0,x1};
    for (int pd=0; pd<2; ++pd) {
        const auto & w = *weights[pd];
        const auto a = w.info();
        if (!w.impl_->ready || a.abi_version != 1 || a.device != pd || a.k != k || a.n != n ||
                (work[pd]->id >= 0 && work[pd]->id != pd) ||
                q6_overlap(inputs[pd],xb,w.data(),a.payload_bytes)) return 0;
    }
    if (q6_overlap(y,2*bytes,w0.data(),w0.info().payload_bytes) || q6_overlap(y,2*bytes,x0,xb)) return 0;

    const auto timestamp = [&]() -> uint64_t {
        if (!timing) return 0;
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    if (timing) { *timing = {}; timing->begin_ns = timestamp(); }
    for (int pd=0; pd<2; ++pd) {
        auto & s = *work[pd];
        s.id = pd;
        select(pd);
        native_reserve(s.quant,pd,(size_t)(k/16)*1024);
        native_reserve(s.scales,pd,(k/16)*4);
        if (pd == 1) native_reserve(s.shard_output,pd,bytes);
        load_kernel(s.programs[0],"kunlun-q6-quant",pd,false);
        load_kernel(s.programs[3],"kunlun-q6-onchip",pd,true);
    }
    if (timing) timing->prepared_ns = timestamp();
    void * remote_output = work[1]->shard_output->data();
    int error = 0;
    for (int pd=0; pd<2; ++pd) {
        auto & s = *work[pd];
        const auto & w = *weights[pd];
        const auto * weight = (const char *) w.data() + (size_t) pd*half*k;
        const auto * scale = (const char *) w.data() + w.info().scale_offset + (size_t) pd*half*(k/16)*sizeof(float);
        const void * x = inputs[pd];
        void * out = pd == 0 ? y : remote_output;
        void * q = s.quant->data();
        void * as = s.scales->data();
        select(pd);
        const uint64_t quant_begin = timestamp();
        error = q6_pair_launch(s.programs[0],{native_arg(q),native_arg(as),native_arg(x),native_arg(k),native_arg(dtype)},16);
        if (timing) timing->quant_enqueue_ns[pd] = timestamp() - quant_begin;
        if (error) break;
        const uint64_t sdnn_begin = timestamp();
        error = q6_pair_launch(s.programs[3],{native_arg(out),native_arg(weight),native_arg(q),native_arg(k),
                                            native_arg(half),native_arg(scale),native_arg(as)},8);
        if (timing) { timing->submit_ns[pd] = timestamp(); timing->sdnn_enqueue_ns[pd] = timing->submit_ns[pd] - sdnn_begin; }
        if (error) break;
    }
    // Drain both devices even if one of the two launches returns an error.
    int drain_error = 0;
    for (int pd=0; pd<2; ++pd) {
        select(pd);
        const uint64_t begin = timestamp();
        const int status = xpu_wait();
        if (timing) { timing->complete_ns[pd] = timestamp(); timing->wait_ns[pd] = timing->complete_ns[pd] - begin; }
        if (status) drain_error = status;
    }
    // A failed wait leaves ownership unknown. Use the existing fatal runtime contract.
    check(drain_error);
    if (error) return -1;

    select(0);
    void * target = (char *) y + bytes;
    kv_max_invalidate(0,target,bytes);
    if (timing) timing->peer_begin_ns = timestamp();
    const int copy_status = xpu_memcpy_peer(0,target,1,remote_output,bytes);
    const int wait_status = xpu_wait();
    if (timing) timing->peer_end_ns = timestamp();
    if (copy_status || wait_status) {
        select(1);
        const int source_wait = xpu_wait();
        check(wait_status);
        check(source_wait);
        return -1;
    }
    ++peer_copies;
    peer_bytes += bytes;
    ++work[0]->stats.calls;
    ++work[1]->stats.calls;
    return 1;
}
