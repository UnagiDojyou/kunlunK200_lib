CommonQ4TPStatistics CommonQ4Workspace::tp_statistics() const {
    std::lock_guard<std::recursive_mutex> lock(mutex);return impl_->tp_stats;
}
namespace {
uint64_t tp_tick() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
void tp_quant_wait(CommonQ4Workspace::Impl & s) {
    select(s.id);uint64_t begin=tp_tick();check(xpu_wait());s.tp_stats.wait_ns+=tp_tick()-begin;
}
// Both sources are completed HBM buffers on PD0/PD1. Preserve local cache metadata.
bool tp_merge_absmax(Kernel::Impl &program,std::unique_ptr<Buffer> &receive,std::unique_ptr<Buffer> &flags,
        void *p0,void *p1,int rows,int chunks,int stride,const std::function<void()> &wait) {
    const size_t bytes=aligned((size_t)rows*stride*4);
    native_reserve(receive,0,bytes);native_reserve(flags,0,256);
    peer_completed(0,receive->data(),1,p1,bytes);
    select(0);void *other=receive->data(),*status=flags->data();
    native_launch(program,{native_arg(p0),native_arg(other),native_arg(status),native_arg(rows),native_arg(chunks),native_arg(stride)},16);
    wait();int bad[64]={};check(xpu_memcpy(bad,status,sizeof(bad),XPU_DEVICE_TO_HOST));wait();
    for(int v:bad)if(v)return false;
    peer_completed(1,p1,0,p0,bytes);return true;
}
bool tp_share_absmax(CommonQ4Workspace::Impl &a,CommonQ4Workspace::Impl &b,void *p0,void *p1,int rows,int chunks,int stride) {
    const uint64_t begin=tp_tick();select(a.id);
    bool ok=tp_merge_absmax(common_program(a,41,"kunlun-tp-absmax"),a.tp_peer,a.tp_status,p0,p1,rows,chunks,stride,[&](){tp_quant_wait(a);});
    if(ok){++a.tp_stats.max_collectives;a.tp_stats.peer_copies+=2;a.tp_stats.peer_bytes+=2*aligned((size_t)rows*stride*4);}
    a.tp_stats.max_ns+=tp_tick()-begin;return ok;
}
// The K200's XRE 4.33 peer and D2H APIs return after the transfer completes.
bool tp_wo_synchronous_copy() {
    static const bool supported=[] {
        uint32_t rmajor=0,rminor=0,dmajor=0,dminor=0;
        uint64_t models[2]{};
        return xpu_get_runtime_version(&rmajor,&rminor)==0 && rmajor==4 && rminor==33 &&
            xpu_get_driver_version(&dmajor,&dminor)==0 && dmajor==4 && dminor==33 &&
            xpu_device_get_attr(&models[0],XPUATTR_MODEL,0)==0 && models[0]==K200 &&
            xpu_device_get_attr(&models[1],XPUATTR_MODEL,1)==0 && models[1]==K200;
    }();
    return supported;
}

bool tp_quant_launch(Kernel::Impl & program,std::initializer_list<NativeArgument> args,
        bool checked,bool packed) {
    if(!checked) {
        native_launch(program,args,16);
        return true;
    }
    size_t offset=0;
    if(packed) {
        // XRE copies argument bytes into its own launch packet. These kernels use
        // consecutive pointer and I32 fields with no padding between arguments.
        unsigned char bytes[80];
        for(auto arg:args) {
            if(arg.size%4 || arg.size>sizeof(bytes)-offset)return false;
            std::memcpy(bytes+offset,arg.value,arg.size);
            offset+=arg.size;
        }
        if(xpu_launch_argument_set(bytes,offset,0))return false;
    }else{
        for(auto arg:args) {
            if(xpu_launch_argument_set(arg.value,arg.size,offset))return false;
            offset+=arg.size;
        }
    }
    return xpu_launch_config(4,16)==0 && xpu_launch_async(program.fn)==0;
}

// Exchange completed peak arrays or WO inputs using the same PD-owned scratch.
bool tp_exchange_completed(CommonQ4Workspace::Impl &a,CommonQ4Workspace::Impl &b,
        const void *p0,const void *p1,int stride,bool synchronous=false) {
    const uint64_t begin=tp_tick();
    const size_t bytes=aligned((size_t)stride*4);
    for (auto *s:{&a,&b}) {
        select(s->id);
        native_reserve(s->tp_peer,s->id,bytes);
    }
    if(synchronous) {
        const auto copy_completed=[&](int to,void * target,int from,const void * source) {
            select(to);
            kv_max_invalidate(to,target,bytes);
            if(xpu_memcpy_peer(to,target,from,source,bytes))return false;
            ++peer_copies;
            peer_bytes+=bytes;
            return true;
        };
        if(!copy_completed(0,a.tp_peer->data(),1,p1) || !copy_completed(1,b.tp_peer->data(),0,p0))return false;
    }else{
        peer_completed(0,a.tp_peer->data(),1,p1,bytes);
        peer_completed(1,b.tp_peer->data(),0,p0,bytes);
    }
    ++a.tp_stats.max_collectives;
    a.tp_stats.peer_copies+=2;
    a.tp_stats.peer_bytes+=2*bytes;
    a.tp_stats.max_ns+=tp_tick()-begin;
    return true;
}
bool tp_quant_status(CommonQ4Workspace::Impl &s,bool synchronous=false) {
    const uint64_t begin=tp_tick();
    unsigned int bad=~0u;
    select(s.id);
    const int result=xpu_memcpy(&bad,s.tp_status->data(),sizeof(bad),XPU_DEVICE_TO_HOST);
    if(!synchronous) {
        check(result);
        tp_quant_wait(s);
    }
    s.tp_stats.max_ns+=tp_tick()-begin;
    return result==0 && bad==0;
}
}

bool common_q4_tp_prepare_global(CommonQ4Workspace & work0,CommonQ4Workspace & work1,
        const void * input0,const void * input1,int k,int m,uint64_t generation,
        CommonQ4ActivationLease & lease0,CommonQ4ActivationLease & lease1,bool parallel) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    const char * attn=kunlun_default_on("GGML_KUNLUN_TP_ATTENTION");
    bool wo=attn && !std::strcmp(attn,"1") && k==2048;
    if(&work0==&work1 || &lease0==&lease1 || !generation || (k!=5504 && !wo) || (m!=1 && m!=128) ||
       !input0 || !input1 || (uintptr_t)input0%4 || (uintptr_t)input1%4)return false;
    CommonQ4Workspace::Impl * states[2]={work0.impl_,work1.impl_};
    const char * opt=kunlun_default_on("GGML_KUNLUN_TP_QUANT_PREP");
    const bool optimize=opt && !std::strcmp(opt,"1");
    const char * max_opt=kunlun_default_on("GGML_KUNLUN_TP_MAX_FUSED");
    const bool fused_max=m==1 && parallel && max_opt && !std::strcmp(max_opt,"1");
    const auto input_addresses=reinterpret_cast<uintptr_t>(input0)|reinterpret_cast<uintptr_t>(input1);
    const bool wo_share=wo && fused_max && optimize && (input_addresses&255)==0 &&
        common_wo_option("GGML_KUNLUN_TP_WO_INPUT_SHARE",true);
    const bool packed=wo_share && common_wo_option("GGML_KUNLUN_TP_WO_PACK_ARGS");
    const bool synchronous=wo_share && common_wo_option("GGML_KUNLUN_TP_WO_SYNC_COPY") && tp_wo_synchronous_copy();
    const bool checked=packed || synchronous;
    const char * down_opt=kunlun_default_on("GGML_KUNLUN_TP_DOWN_DIRECT_M1");
    const bool direct_down=m==1 && k==5504 && down_opt && !std::strcmp(down_opt,"1");
    const void * input[2]={};
    bool direct[2]={};
    CommonQ4ActivationLease * leases[2]={&lease0,&lease1};const void * inputs[2]={input0,input1};
    for(int d=0;d<2;++d){auto & s=*states[d];
        if((s.id!=-1 && s.id!=d) || s.pending || s.preparing || s.post_pending || s.crop_pending || s.crop_running)return false;
        if((uintptr_t)inputs[d]+(size_t)k*m*4<(uintptr_t)inputs[d])return false;
        for(auto * buf:{s.input.get(),s.activation->quant.get(),s.activation->scale.get()})if(buf) {
            uintptr_t a=(uintptr_t)inputs[d],b=(uintptr_t)buf->data();if(a<b+buf->size() && b<a+(size_t)k*m*4)return false;
        }
        if (optimize || fused_max || direct_down) for (auto * buf : {s.peaks.get(),s.peak1.get(),s.rest.get(),s.tp_peer.get(),s.tp_status.get()}) if (buf) {
            const uintptr_t a=(uintptr_t)inputs[d], b=(uintptr_t)buf->data();
            if (a < b+buf->size() && b < a+(size_t)k*m*4) return false;
        }
    }
    const uint64_t begin=tp_tick();const int kp=k==5504?5632:2048,dtype=0,chunks=kp/64,stride=(chunks+15)/16*16;
    const size_t peak_bytes=std::max<size_t>(2048,aligned((size_t)m*stride*4));
    for(int d=0;d<2;++d) {
        auto & s=*states[d];auto & a=*s.activation;auto & l=*leases[d]->impl_;
        leases[d]->reset();common_invalidate(a);s.last_components=0;s.id=d;select(d);
        direct[d]=((optimize && (wo || m==128)) || direct_down) && (uintptr_t)inputs[d]%64==0;
        if (!direct[d]) native_reserve(s.input,d,(size_t)m*kp*4);
        native_reserve(s.peaks,d,peak_bytes);
        native_reserve(a.quant,d,m==1?(size_t)4*kp+1024:(size_t)m*kp+1024);
        native_reserve(a.scale,d,m==1?64:(size_t)m*64);
        if(m==1){native_reserve(s.rest,d,kp*4);native_reserve(s.peak1,d,peak_bytes);}
        s.preparing=true;
        input[d]=direct[d]?inputs[d]:s.input->data();
        if (!direct[d]) common_rows(s,s.input->data(),inputs[d],k,m,k,kp,m,4,parallel);
        l.storage=s.activation;l.epoch=a.epoch;l.generation=generation;l.id=d;l.k=k;l.m=m;
        l.type=InputType::f32;l.input=inputs[d];l.input_bytes=(size_t)m*k*4;
    }
    const auto reject=[&](){
        for(auto * s:states){
            select(s->id);
            check(xpu_wait());
            s->preparing=false;
            common_invalidate(*s->activation);
        }
        for(auto * l:leases)l->reset();
        return false;
    };
    if (wo_share) {
        for(auto * s:states)tp_quant_wait(*s);
        if(!tp_exchange_completed(*states[0],*states[1],input[0],input[1],k,synchronous))return reject();
    }
    // Padding and local-peak kernels use the same ordered CL queue on each PD.
    if (!optimize || !parallel) for(auto * s:states)tp_quant_wait(*s);
    for(auto * s:states) {
        select(s->id);void * peaks=s->peaks->data();const void * x=input[s->id];
        if(m==1) {
            if(wo_share) {
                const void * remote=s->tp_peer->data();
                auto & maximum=common_program(*s,48,"kunlun-tp-wo-shared-max");
                if(!tp_quant_launch(maximum,{native_arg(peaks),native_arg(x),native_arg(remote)},checked,packed))return reject();
            }else if(direct_down && direct[s->id]) {
                native_launch(common_program(*s,46,"kunlun-tp-quant-max-m1"),{native_arg(peaks),native_arg(x),native_arg(kp),native_arg(dtype),native_arg(k)},16);
            }else native_launch(common_program(*s,34,"kunlun-common-max-exact"),{native_arg(peaks),native_arg(x),native_arg(kp),native_arg(dtype)},16);
        }
        else {
            const bool logical=direct[s->id] && !wo;
            native_launch(common_program(*s,logical?42:5,logical?"kunlun-tp-quant-partial":"kunlun-common-quant-partial"),{native_arg(peaks),native_arg(x),native_arg(m),native_arg(kp),native_arg(dtype)},16);
        }
        if(!parallel)tp_quant_wait(*s);
    }
    if(!wo_share)for(auto * s:states)tp_quant_wait(*s);
    if(!wo_share) {
        if(fused_max)tp_exchange_completed(*states[0],*states[1],states[0]->peaks->data(),states[1]->peaks->data(),stride);
        else if(!tp_share_absmax(*states[0],*states[1],states[0]->peaks->data(),states[1]->peaks->data(),m,chunks,stride))return reject();
    }
    for(auto * s:states) {
        select(s->id);void * q=s->activation->quant->data(),*scale=s->activation->scale->data(),*peaks=s->peaks->data();const void * x=input[s->id];
        if(m==1){void * residual=s->rest->data(),*p1=s->peak1->data();
            const bool pad_in_kernel=direct_down && direct[s->id];
            if(fused_max || pad_in_kernel) {
                void * peer=fused_max?s->tp_peer->data():nullptr;
                const int logical_k=pad_in_kernel?k:kp;
                auto & main=common_program(*s,wo_share?49:44,wo_share?"kunlun-tp-wo-shared-main":"kunlun-tp-quant-main-m1");
                if(!tp_quant_launch(main,{native_arg(q),native_arg(residual),native_arg(p1),native_arg(x),native_arg(peaks),
                    native_arg(peer),native_arg(scale),native_arg(kp),native_arg(dtype),native_arg(logical_k)},checked,packed))return reject();
            }else native_launch(common_program(*s,35,"kunlun-common-main-exact"),{native_arg(q),native_arg(residual),native_arg(p1),native_arg(x),native_arg(peaks),native_arg(kp),native_arg(dtype)},16);
        }else{int mode=1;
            const bool logical=direct[s->id] && !wo;
            native_launch(common_program(*s,logical?43:6,logical?"kunlun-tp-quant-row":"kunlun-common-quant-row-fused"),{native_arg(q),native_arg(scale),native_arg(x),native_arg(peaks),native_arg(m),native_arg(kp),native_arg(dtype),native_arg(mode)},16);
        }
        if(!parallel)tp_quant_wait(*s);
    }
    if(!wo_share)for(auto * s:states)tp_quant_wait(*s);
    if(m==1) {
        if(fused_max) {
            select(0);
            native_reserve(states[0]->tp_status,0,256);
            if(!wo_share)tp_exchange_completed(*states[0],*states[1],states[0]->peak1->data(),states[1]->peak1->data(),stride);
        }else if(!tp_share_absmax(*states[0],*states[1],states[0]->peak1->data(),states[1]->peak1->data(),1,chunks,stride))return reject();
        for(auto * s:states){select(s->id);void * q=s->activation->quant->data(),*params=s->activation->scale->data(),*r=s->rest->data(),*p0=s->peaks->data(),*p1=s->peak1->data();
            if(fused_max) {
                // Both PDs validate the same peak arrays. PD0 reports the final status.
                void * peer=s->tp_peer->data(),*status=s->id==0?s->tp_status->data():nullptr;
                auto & residual=common_program(*s,wo_share?50:45,wo_share?"kunlun-tp-wo-shared-residual":"kunlun-tp-quant-residual-m1");
                if(!tp_quant_launch(residual,{native_arg(q),native_arg(params),native_arg(r),native_arg(p0),native_arg(p1),
                    native_arg(peer),native_arg(status),native_arg(kp)},checked,packed))return reject();
            }else native_launch(common_program(*s,36,"kunlun-common-residual-exact"),{native_arg(q),native_arg(params),native_arg(r),native_arg(p0),native_arg(p1),native_arg(kp)},16);
            if(!parallel)tp_quant_wait(*s);
        }
        for(auto * s:states)tp_quant_wait(*s);
        if(fused_max && !tp_quant_status(*states[0],synchronous))return reject();
    }
    for(int d=0;d<2;++d){auto & s=*states[d];auto & a=*s.activation;auto & l=*leases[d]->impl_;
        s.preparing=false;a.valid=true;++a.stats.prepare_calls;++a.stats.quantized_chunks;
        a.stats.quant_kernel_launches+=m==1?3:2;
        l.chunks.push_back({a.quant->data(),a.scale->data(),0,m,m,kp,m==1?2:1,m==1?(size_t)4*kp+1024:(size_t)m*kp,m==1?64:(size_t)m*64});
        s.last_values=a.quant->data();s.last_scales=a.scale->data();s.last_rows=m;s.last_kp=kp;s.last_components=m==1?2:1;
        ++s.tp_stats.prepares;s.tp_stats.prepare_ns+=tp_tick()-begin;
    }
    return true;
}
