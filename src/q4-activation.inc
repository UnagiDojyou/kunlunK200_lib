CommonQ4ActivationLease::CommonQ4ActivationLease(): impl_(new Impl{}) {}
CommonQ4ActivationLease::~CommonQ4ActivationLease() { reset(); delete impl_; }
void CommonQ4ActivationLease::reset() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto & a=*impl_;
    if (a.storage && a.storage->valid && a.epoch==a.storage->epoch) common_invalidate(*a.storage);
    a.storage.reset();a.chunks.clear();a.epoch=a.generation=0;
}
bool CommonQ4ActivationLease::matches(int id, int k, int m, InputType type, uint64_t generation) const {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    const auto & a=*impl_;
    return generation && a.storage && a.storage->valid && a.epoch==a.storage->epoch &&
        a.generation==generation && a.id==id && a.k==k && a.m==m && a.type==type;
}
size_t CommonQ4ActivationLease::chunks() const {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    const auto & a=*impl_;
    return a.storage && a.storage->valid && a.epoch==a.storage->epoch ? a.chunks.size() : 0;
}
CommonQ4ActivationChunk CommonQ4ActivationLease::chunk(size_t index) const {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return index<chunks() ? impl_->chunks[index] : CommonQ4ActivationChunk{};
}
CommonQ4ActivationStatistics CommonQ4Workspace::activation_statistics() const {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    const auto & a=*impl_->activation;auto stats=a.stats;
    for (auto * p : {a.quant.get(),a.scale.get()}) if (p) stats.hbm_activation_bytes+=std::max<size_t>(p->size(),256);
    return stats;
}

bool common_q4_prepare(CommonQ4Workspace & work, int id, const void * input, int k, int m,
                       InputType type, uint64_t generation, CommonQ4ActivationLease & lease) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto & s=*work.impl_;auto & a=*s.activation;
    auto reject=[&]() { ++a.stats.rejected_calls;return false; };
    if (id<0 || id>1 || !input || !generation || !common_q4_supported(k,1) || m<=0 ||
        (size_t)m*k>INT_MAX/4 || (type!=InputType::f32 && type!=InputType::f16) || (s.id!=-1 && s.id!=id)) return reject();
    int bytes=type==InputType::f16 ? 2 : 4,dtype=bytes==2;
    if ((uintptr_t)input%bytes || (uintptr_t)input+(size_t)m*k*bytes<(uintptr_t)input) return reject();
    uintptr_t xp=(uintptr_t)input;size_t input_bytes=(size_t)m*k*bytes;
    // A raw borrowed quantization view is not a valid source for overwriting
    // its own workspace. Reject before reset/reserve can retire that storage.
    for(auto * b:{a.quant.get(),a.scale.get()})if(b) {
        uintptr_t bp=(uintptr_t)b->data();
        if(xp<bp+b->size() && bp<xp+input_bytes)return reject();
    }
    int kp=common_q4::describe(k,1,common_q4::Layout::Micro32Bank,common_q4::ScaleOrder::KMajor).kp;
    size_t count=((size_t)m+127)/128;
    size_t qstride=m==1 ? (size_t)4*kp+1024 : (size_t)128*kp+1024;
    size_t sstride=m==1 ? 64 : 128*64;
    if (count*qstride>16*1024*1024) return reject();
    uint64_t begin=common_tick(a.profile);
    lease.reset();common_invalidate(a);s.last_components=0;s.id=id;select(id);
    native_reserve(a.quant,id,count*qstride);native_reserve(a.scale,id,count*sstride);
    auto & l=*lease.impl_;l.storage=s.activation;l.epoch=a.epoch;l.generation=generation;
    l.id=id;l.k=k;l.m=m;l.type=type;l.input=input;l.input_bytes=(size_t)m*k*bytes;
    size_t index=0;
    for (int first=0;first<m;++index) {
        int rows=std::min(128,m-first),padded=m==1?1:rows<=8?8:rows<=32?32:rows<=64?64:128;
        const void * x=(const char *)input+(size_t)first*k*bytes;
        if (k!=kp || rows!=padded || (uintptr_t)x%64) {
            native_reserve(s.input,id,(size_t)padded*kp*bytes);
            common_rows(s,s.input->data(),x,k,rows,k,kp,padded,bytes);x=s.input->data();
        }
        void * q=(char *)a.quant->data()+index*qstride,* scale=(char *)a.scale->data()+index*sstride;
        if (m==1) common_quant_gemv(s,x,kp,dtype,q,scale);
        else common_quant_gemm(s,x,padded,kp,dtype,q,scale);
        int defined_rows=m==1 ? 1 : (padded+31)/32*32;
        l.chunks.push_back({q,scale,first,rows,padded,kp,m==1?2:1,
            m==1?qstride:(size_t)defined_rows*kp,m==1?sstride:(size_t)defined_rows*64});
        s.last_values=q;s.last_scales=scale;s.last_rows=rows;s.last_kp=kp;s.last_components=m==1?2:1;
        first+=rows;
    }
    a.valid=true;++a.stats.prepare_calls;
    if (a.profile) a.stats.preparation_ns+=common_tick(true)-begin;
    return true;
}

bool common_q4_matmul_shared(CommonQ4Workspace & work, int id, void * y, const CommonQ4Weight & weight,
                             const CommonQ4ActivationLease & lease, uint64_t generation) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto & s=*work.impl_;const auto & l=*lease.impl_;const auto & w=*weight.impl_;auto a=w.info;
    auto reject=[&]() { ++s.activation->stats.rejected_calls;return false; };
    if (l.storage!=s.activation || !lease.matches(id,a.k,l.m,l.type,generation) || !w.ready || id!=a.device || s.id!=id || !y) return reject();
    auto e=common_q4::describe(a.k,a.n,common_q4::Layout::Micro32Bank,common_q4::ScaleOrder::KMajor);
    if (a.abi_version!=1 || a.quant_type!=2 || a.layout!=1 || a.scale_order!=1 || a.kp!=e.kp || a.np!=e.np ||
        a.scale_offset!=e.scale_offset || a.payload_bytes!=e.payload_bytes || (size_t)l.m*a.n>INT_MAX/4) return reject();
    uintptr_t yp=(uintptr_t)y,wp=(uintptr_t)w.storage;size_t ys=(size_t)l.m*a.n*4;
    auto overlap=[&](const void * p,size_t n) { uintptr_t v=(uintptr_t)p;return yp<v+n && v<yp+ys; };
    if (yp%4 || yp+ys<yp || wp%256 || wp+a.payload_bytes<wp || overlap(l.input,l.input_bytes) ||
        overlap(w.storage,a.payload_bytes) ||
        ((uintptr_t)l.input<wp+a.payload_bytes && wp<(uintptr_t)l.input+l.input_bytes) || overlap(s.activation->quant->data(),s.activation->quant->size()) ||
        overlap(s.activation->scale->data(),s.activation->scale->size())) return reject();
    select(id);
    if (!common_execute(s,id,y,a,w,l.input,l.m,l.type,&l)) return reject();
    ++s.activation->stats.prepared_matmuls;s.activation->stats.prepared_chunks+=l.chunks.size();
    return true;
}
