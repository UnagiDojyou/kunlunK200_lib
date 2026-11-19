CommonQ4PairStatus common_q4_pair_last_status(const CommonQ4Workspace & work) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return work.impl_->pair_status;
}
CommonQ4PairStatistics common_q4_pair_statistics(const CommonQ4Workspace & work) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return work.impl_->pair_stats;
}
bool common_q4_matmul_pair(CommonQ4Workspace & work,int id,void * y0,void * y1,
        const CommonQ4Weight & weight0,const CommonQ4Weight & weight1,
        const CommonQ4ActivationLease & lease,uint64_t generation) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto & s=*work.impl_;const auto & l=*lease.impl_;
    const auto & w0=*weight0.impl_;const auto & w1=*weight1.impl_;auto a=w0.info,b=w1.info;
    const char * row_mode=std::getenv("GGML_KUNLUN_ROW_PAIR");
    const bool row=s.submit_only && (s.tp_mode || (!row_mode || !std::strcmp(row_mode,"1")));
    if(s.pending || s.preparing || s.post_pending || s.crop_pending || s.crop_running)return false;
    auto reject=[&](CommonQ4PairStatus reason=CommonQ4PairStatus::InvalidInput){++s.pair_stats.declined;s.pair_status=reason;return false;};
    const char *flag=std::getenv("GGML_KUNLUN_Q4_GATE_UP_PAIR");
    if(flag && std::strcmp(flag,"auto") && std::strcmp(flag,"1"))return reject(CommonQ4PairStatus::Disabled);
    const int gate=s.tp_pair_glu_gate;
    const bool glu=gate>=0;
    if(id<0 || id>1 || s.id!=id || !y0 || (!glu && !y1) ||
       !w0.ready || !w1.ready || a.device!=id || b.device!=id || a.k!=b.k || a.n!=b.n ||
       !((!row && ((a.k==4096 && a.n==11008) || (a.k==2048 && a.n==5632))) ||
         (row && ((a.k==4096 && a.n==5504 && a.np==5632) || (a.k==2048 && a.n==2816)))))return reject();
    if(l.storage!=s.activation || l.m!=1 || l.type!=InputType::f32 ||
       !lease.matches(id,a.k,1,InputType::f32,generation) || l.chunks.size()!=1)return reject(CommonQ4PairStatus::Lease);
    for(const char *name:{"KUNLUN_Q4_MAC_DMA","KUNLUN_Q4_DMA_ADDR32"}){
        const char *value=std::getenv(name);if(value && std::strcmp(value,"1"))return reject(CommonQ4PairStatus::Configuration);
    }
    if(a.k==4096){const char *value=std::getenv("KUNLUN_Q4_NMAJOR");if(value && std::strcmp(value,"1"))return reject(CommonQ4PairStatus::Configuration);}
    auto expected=common_q4::describe(a.k,a.n,common_q4::Layout::Micro32Bank,common_q4::ScaleOrder::KMajor);
    for(const auto &v:{a,b})if(v.abi_version!=1 || v.quant_type!=2 || v.layout!=1 || v.scale_order!=1 ||
        v.k!=v.kp || (!row && v.n!=v.np) || v.kp!=expected.kp || v.np!=expected.np ||
        v.scale_offset!=expected.scale_offset || v.payload_bytes!=expected.payload_bytes)return reject(CommonQ4PairStatus::WeightAbi);
    struct Range{uintptr_t p;size_t bytes;};
    Range outputs[]={{(uintptr_t)y0,(size_t)(glu?a.n:a.np)*4},{(uintptr_t)y1,glu?0:(size_t)a.np*4}};
    Range inputs[]={{(uintptr_t)w0.storage,a.payload_bytes},{(uintptr_t)w1.storage,b.payload_bytes},
        {(uintptr_t)l.input,l.input_bytes},{(uintptr_t)s.activation->quant->data(),s.activation->quant->size()},
        {(uintptr_t)s.activation->scale->data(),s.activation->scale->size()}};
    auto overlap=[](const Range &x,const Range &y){return x.bytes && y.bytes && x.p<y.p+y.bytes && y.p<x.p+x.bytes;};
    for(auto x:outputs)if(x.p%64 || x.bytes>UINTPTR_MAX-x.p)return reject(CommonQ4PairStatus::Alias);
    for(auto x:inputs)if(!x.p || x.bytes>UINTPTR_MAX-x.p)return reject(CommonQ4PairStatus::Alias);
    if(overlap(outputs[0],outputs[1]) || w0.storage==w1.storage || (uintptr_t)w0.storage%256 || (uintptr_t)w1.storage%256 ||
       overlap(inputs[0],inputs[2]) || overlap(inputs[1],inputs[2]))return reject(CommonQ4PairStatus::Alias);
    for(auto x:outputs)for(auto y:inputs)if(overlap(x,y))return reject(CommonQ4PairStatus::Alias);
    const auto &chunk=l.chunks[0];
    if(chunk.kp!=a.k || chunk.rows!=1 || chunk.padded_rows!=1 || chunk.components!=2)return reject(CommonQ4PairStatus::Lease);
    const int k=a.k,n=row?a.np:a.n,batch=1,first=0,components=2,bits=8,pipelined=1,fold=0,mode=1;
    const size_t partial_bytes=(size_t)n*(k/32)*4;
    if(partial_bytes>8*1024*1024)return reject(CommonQ4PairStatus::Workspace);
    select(id);
    if(!s.l3){s.l3=native_l3[id].lock();if(!s.l3){s.l3=std::make_shared<NativeL3>(id);native_l3[id]=s.l3;}}
    if (glu && (overlap(outputs[0],{(uintptr_t)s.l3->pointer,s.l3->bytes}) ||
            (s.output && overlap(outputs[0],{(uintptr_t)s.output->data(),s.output->size()})) ||
            (s.input && overlap(outputs[0],{(uintptr_t)s.input->data(),s.input->size()}))))
        return reject(CommonQ4PairStatus::Alias);
    if(s.l3_unavailable || !s.l3->reserve(partial_bytes*2))return reject(CommonQ4PairStatus::Workspace);
    void *p0=s.l3->pointer,*p1=(char*)p0+partial_bytes;
    const void *q=chunk.values,*params=chunk.scales,*w=w0.storage,*u=w1.storage;
    const void *scale0=(const char*)w+a.scale_offset,*scale1=(const char*)u+b.scale_offset;
    float zero=0,one=1;bool nmajor=k==4096;
    const char *preaddr_mode=std::getenv("KUNLUN_Q4_PAIR_PREADDR");
    const bool preaddr=(!preaddr_mode || !std::strcmp(preaddr_mode,"1"));
    const char *dma_mode=std::getenv("KUNLUN_Q4_PAIR_DMA_CONST");
    const bool dma_const=(!dma_mode || !std::strcmp(dma_mode,"1"));
    const int program=row?(k==4096?39:40):dma_const?(preaddr?(nmajor?31:32):(nmajor?29:30)):(preaddr?(nmajor?27:28):(nmajor?24:25));
    const char *name=row?(k==4096?"kunlun-row-pair-7b":"kunlun-row-pair-tiny"):dma_const?
        (preaddr?(nmajor?"kunlun-common-gemv-pair-dma-const-preaddr-nmajor":"kunlun-common-gemv-pair-dma-const-preaddr-stream"):
                 (nmajor?"kunlun-common-gemv-pair-dma-const-nmajor":"kunlun-common-gemv-pair-dma-const-stream")):
        (preaddr?(nmajor?"kunlun-common-gemv-pair-preaddr-nmajor":"kunlun-common-gemv-pair-preaddr-stream"):
                 (nmajor?"kunlun-common-gemv-pair-nmajor":"kunlun-common-gemv-pair-stream"));
    const char *sum_mode=std::getenv("KUNLUN_Q4_PAIR_SUM_FAST");
    const bool sum_fast=(!sum_mode || !std::strcmp(sum_mode,"1"));
    auto & sum=glu ? common_program(s,51,"kunlun-tp-pair-sum-glu") :
            common_program(s,sum_fast?33:26,sum_fast?"kunlun-common-sum-pair-fast":"kunlun-common-sum-pair");
    auto finish=[&,y0,p0,scale0,k,n,batch,first,mode,y1,p1,gate]() {
        select(s.id);check(xpu_wait());
        if (gate>=0) {
            const void * g=gate==0?p0:p1, * u=gate==0?p1:p0;
            if (native_launch_status(sum,{native_arg(y0),native_arg(g),native_arg(u)},16)!=0) {
                s.pair_status=CommonQ4PairStatus::PostFailed;
            }
        } else {
            native_launch(sum,{native_arg(y0),native_arg(p0),native_arg(scale0),native_arg(k),native_arg(n),native_arg(batch),native_arg(first),native_arg(n),native_arg(mode),native_arg(y1),native_arg(p1)},16);
        }
        // Drain even when the fused launch reports an error.
        if(s.post_only)s.post_pending=true;else check(xpu_wait());
    };
    native_launch(common_program(s,program,name,true),
        {native_arg(p0),native_arg(w),native_arg(q),native_arg(k),native_arg(n),native_arg(batch),native_arg(first),native_arg(n),
         native_arg(components),native_arg(bits),native_arg(zero),native_arg(zero),native_arg(one),native_arg(pipelined),native_arg(scale0),native_arg(fold),native_arg(params),native_arg(u),native_arg(scale1),native_arg(p1)},8);
    if(s.submit_only)s.pending=finish;else finish();
    s.last_values=q;s.last_scales=params;s.last_components=2;s.last_rows=1;s.last_kp=k;
    s.stats.paths[0]+=2;s.stats.calls+=2;s.stats.chunks+=2;matmuls[id]+=2;++s.mac_stats.stream_sd_calls;
    s.mac_stats.resident_l1d_bytes=std::max(s.mac_stats.resident_l1d_bytes,(size_t)(k/32)*4096);
    s.activation->stats.prepared_matmuls+=2;s.activation->stats.prepared_chunks+=2;
    s.pair_status=CommonQ4PairStatus::Success;
    ++s.pair_stats.calls;s.pair_stats.l3_partial_bytes=std::max(s.pair_stats.l3_partial_bytes,partial_bytes*2);
    return true;
}

bool common_q4_submit_pair_shared(CommonQ4Workspace & work,int id,void * y0,void * y1,
        const CommonQ4Weight & w0,const CommonQ4Weight & w1,const CommonQ4ActivationLease & lease,uint64_t generation) {
    std::lock_guard<std::recursive_mutex> lock(mutex);auto & s=*work.impl_;
    if(s.pending || s.preparing || s.post_pending || s.crop_pending || s.crop_running)return false;
    s.submit_only=true;bool ok=common_q4_matmul_pair(work,id,y0,y1,w0,w1,lease,generation);s.submit_only=false;
    require(!ok || bool(s.pending));return ok;
}
