bool common_q4_get_rows(Kernel &kernel,int id,void *output,size_t output_bytes,const CommonQ4Weight &weight,const int32_t *indices,size_t index_bytes,int rows) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto a=weight.info();
    if(a.abi_version!=1 || a.quant_type!=2 || a.layout!=1 || a.scale_order!=1 || !weight.impl_->ready || a.device!=id || id<0 || id>1 || a.k%64 || rows<1 || rows>512)return false;
    size_t ybytes=(size_t)rows*a.k*4, ibytes=(size_t)rows*4;
    if(!output || !indices || (uintptr_t)output%64 || (uintptr_t)indices%4 || output_bytes<ybytes || index_bytes<ibytes)return false;
    auto overlap=[](const void *a,size_t an,const void *b,size_t bn){auto x=(uintptr_t)a,y=(uintptr_t)b;require(x<=UINTPTR_MAX-an && y<=UINTPTR_MAX-bn);return x<y+bn && y<x+an;};
    if(overlap(output,ybytes,indices,ibytes) || overlap(output,ybytes,weight.data(),a.payload_bytes) || overlap(indices,ibytes,weight.data(),a.payload_bytes))return false;
    auto &parent=*kernel.impl_;
    require(parent.id<0 || parent.id==id);parent.id=id;select(id);
    if(!parent.get_rows)parent.get_rows=new Kernel::Impl{};
    auto &p=*parent.get_rows;require(p.id<0 || p.id==id);
    load_kernel(p,"kunlun-get-rows",id);
    if(!p.scratch){p.scratch_size=256;p.scratch=allocate(id,p.scratch_size);require(p.scratch);}
    int chunk=rows>=128 && a.k%256==0?256:64;
    if(const char *mode=std::getenv("KUNLUN_GET_ROWS_TILE")) {
        if(!std::strcmp(mode,"64"))chunk=64;
        else if(!std::strcmp(mode,"256") && a.k%256==0)chunk=256;
    }
    const void *weights=weight.data(),*scales=(const char*)weights+a.scale_offset;void *status=p.scratch;
    native_launch(p,{native_arg(output),native_arg(weights),native_arg(scales),native_arg(indices),native_arg(status),native_arg(a.k),native_arg(a.kp),native_arg(a.np),native_arg(a.n),native_arg(rows),native_arg(chunk)},16);
    check(xpu_wait());
    int flags[64];check(copy(flags,status,sizeof(flags),device_to_host));check(xpu_wait());
    for(int flag:flags)if(flag)return false;
    return true;
}
