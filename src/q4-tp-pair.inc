bool common_q4_tp_submit_pair_glu(CommonQ4Workspace & work,int id,void * y,size_t y_bytes,bool swap,
        const CommonQ4Weight & w0,const CommonQ4Weight & w1,const CommonQ4ActivationLease & lease,uint64_t generation) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto & s=*work.impl_;
    const auto a=w0.info(), b=w1.info();
    const char * sum_mode=std::getenv("KUNLUN_Q4_PAIR_SUM_FAST");
    if ((sum_mode && std::strcmp(sum_mode,"1")) || y_bytes<5504*sizeof(float) ||
        a.k!=4096 || a.n!=5504 || a.np!=5632 || b.k!=4096 || b.n!=5504 || b.np!=5632) return false;
    s.tp_mode=true;
    s.tp_pair_glu_gate=swap?1:0;
    const bool ok=common_q4_submit_pair_shared(work,id,y,nullptr,w0,w1,lease,generation);
    s.tp_pair_glu_gate=-1;
    s.tp_mode=false;
    return ok;
}

bool common_q4_tp_submit_pair_padded(CommonQ4Workspace & work,int id,void * y0,void * y1,
        const CommonQ4Weight & w0,const CommonQ4Weight & w1,const CommonQ4ActivationLease & lease,uint64_t generation) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto & s=*work.impl_;
    const auto a=w0.info(), b=w1.info();
    if (a.k!=4096 || a.n!=5504 || a.np!=5632 || b.k!=4096 || b.n!=5504 || b.np!=5632) return false;
    s.tp_mode=true;
    const bool ok=common_q4_submit_pair_shared(work,id,y0,y1,w0,w1,lease,generation);
    s.tp_mode=false;
    return ok;
}

bool common_q4_tp_submit_pair(CommonQ4Workspace & work,int id,void * y0,void * y1,
        const CommonQ4Weight & w0,const CommonQ4Weight & w1,const CommonQ4ActivationLease & lease,uint64_t generation) {
    std::lock_guard<std::recursive_mutex> lock(mutex);auto & s=*work.impl_;auto a=w0.info(),b=w1.info();
    if(s.pending || s.preparing || s.post_pending || s.crop_pending || s.crop_running ||
       a.k!=4096 || a.n!=5504 || a.np!=5632 || b.k!=a.k || b.n!=a.n || id!=a.device || id!=b.device)return false;
    const size_t bytes=(size_t)a.n*4;uintptr_t out[2]={(uintptr_t)y0,(uintptr_t)y1};const auto & l=*lease.impl_;
    if(!y0 || !y1 || (out[0]|out[1])%4 || out[0]+bytes<out[0] || out[1]+bytes<out[1] ||
       !(out[0]+bytes<=out[1] || out[1]+bytes<=out[0]))return false;
    auto overlap=[&](uintptr_t p,size_t n){for(auto y:out)if(y<p+n && p<y+bytes)return true;return false;};
    if(overlap((uintptr_t)l.input,l.input_bytes) || overlap((uintptr_t)w0.data(),a.payload_bytes) || overlap((uintptr_t)w1.data(),b.payload_bytes))return false;
    for(auto * p:{s.input.get(),s.output.get(),s.activation->quant.get(),s.activation->scale.get()})if(p && overlap((uintptr_t)p->data(),p->size()))return false;
    select(id);native_reserve(s.output,id,(size_t)2*a.np*4);void * p0=s.output->data(),*p1=(float *)p0+a.np;
    s.tp_mode=true;bool ok=common_q4_submit_pair_shared(work,id,p0,p1,w0,w1,lease,generation);s.tp_mode=false;
    if(!ok)return false;
    auto finish=s.pending;const int n=a.n,np=a.np;
    s.pending=[&,finish,y0,y1,p0,p1,n,np](){
        finish();
        auto crop=[&,y0,y1,p0,p1,n,np](){
            common_rows(s,y0,p0,n,1,np,n,1,4,s.crop_only);
            common_rows(s,y1,p1,n,1,np,n,1,4,s.crop_only);
            if(s.crop_only)s.crop_running=true;
        };
        if(s.post_only)s.crop_pending=crop;else crop();
    };
    return true;
}
