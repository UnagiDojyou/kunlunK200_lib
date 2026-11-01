bool get_rows_f32(Kernel &kernel,int id,void *output,size_t output_bytes,const void *input,size_t input_bytes,const int32_t *indices,size_t index_bytes,int k,int n,int rows) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(id<0 || id>1 || k<16 || k>1048576 || k%16 || n<1 || rows<1 || rows>512)return false;
    const size_t row_bytes=(size_t)k*4;
    if((size_t)n>SIZE_MAX/row_bytes || (size_t)rows>SIZE_MAX/row_bytes)return false;
    size_t xbytes=(size_t)n*row_bytes,ybytes=(size_t)rows*row_bytes,ibytes=(size_t)rows*4;
    if(!output || !input || !indices || (uintptr_t)output%64 || (uintptr_t)input%64 || (uintptr_t)indices%4 || output_bytes<ybytes || input_bytes<xbytes || index_bytes<ibytes)return false;
    auto fits=[](const void*p,size_t bytes){return (uintptr_t)p<=UINTPTR_MAX-bytes;};
    if(!fits(output,ybytes)||!fits(input,xbytes)||!fits(indices,ibytes))return false;
    auto overlaps=[](const void*a,size_t an,const void*b,size_t bn){auto x=(uintptr_t)a,y=(uintptr_t)b;return x<y+bn&&y<x+an;};
    if(overlaps(output,ybytes,input,xbytes)||overlaps(output,ybytes,indices,ibytes)||overlaps(input,xbytes,indices,ibytes))return false;
    auto &parent=*kernel.impl_;
    require(parent.id<0 || parent.id==id);parent.id=id;select(id);
    if(!parent.get_rows_f32)parent.get_rows_f32=new Kernel::Impl{};
    auto &p=*parent.get_rows_f32;
    load_kernel(p,"kunlun-get-rows-f32",id);
    if(!p.scratch){p.scratch_size=256;p.scratch=allocate(id,p.scratch_size);require(p.scratch);}
    int chunk=rows==1?64:1024;
    if(const char*s=std::getenv("KUNLUN_GET_ROWS_F32_TILE")) {
        if(!std::strcmp(s,"64"))chunk=64;
        else if(!std::strcmp(s,"1024"))chunk=1024;
    }
    void *status=p.scratch;
    native_launch(p,{native_arg(output),native_arg(input),native_arg(indices),native_arg(status),native_arg(k),native_arg(n),native_arg(rows),native_arg(chunk)},16);
    check(xpu_wait());
    int flags[64];check(copy(flags,status,sizeof(flags),device_to_host));check(xpu_wait());
    for(int flag:flags)if(flag)return false;
    return true;
}
