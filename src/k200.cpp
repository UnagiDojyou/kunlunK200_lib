#include <kunlun/k200.h>
#include <xpu/runtime.h>
#include <xpu/xdnn.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <limits.h>
#include <memory>
#include <chrono>
#include <stdexcept>

namespace kunlun {
namespace xdnn = baidu::xpu::api;
namespace {
std::recursive_mutex mutex;
std::atomic<size_t> allocated[2];
std::atomic<size_t> matmuls[2], peer_copies, peer_bytes;
thread_local int diagnostic_device=-1;
thread_local const char * diagnostic_name=nullptr;
thread_local int64_t diagnostic_shape[4]={};
void diagnostic_print() {
    std::fprintf(stderr,"Kunlun context: pd=%d tensor=%s shape=[%lld,%lld,%lld,%lld]\n",diagnostic_device,diagnostic_name?diagnostic_name:"<outside graph>",
        (long long)diagnostic_shape[0],(long long)diagnostic_shape[1],(long long)diagnostic_shape[2],(long long)diagnostic_shape[3]);
    std::fflush(stderr);
}
void library_anchor() {}
std::string kernel_directory() {
    if (const char * path = std::getenv("KUNLUN_KERNEL_DIR")) return path;
    Dl_info info{};
    require(dladdr(reinterpret_cast<void *>(&library_anchor), &info) != 0);
    char resolved[PATH_MAX];
    require(realpath(info.dli_fname, resolved) != nullptr);
    std::string path(resolved);
    return path.substr(0, path.find_last_of('/')) + "/kunlunk200/kernels";
}
}
std::recursive_mutex & runtime_mutex() { return mutex; }
void diagnostic_tensor(const char * name, const int64_t * dimensions) {
    diagnostic_name=name;for(int i=0;i<4;++i)diagnostic_shape[i]=dimensions?dimensions[i]:0;
}

void check(int rc) {
    if (rc) { diagnostic_print(); std::fprintf(stderr, "Kunlun runtime error %d\n", rc); std::fflush(stderr); std::abort(); }
}
void require(bool ok) {
    if (!ok) { diagnostic_print(); std::fprintf(stderr, "Kunlun invalid argument or unavailable resource\n"); std::fflush(stderr); std::abort(); }
}
static inline void check_at(int status,const char *file,int line,const char *api) {
    if(status){std::fprintf(stderr,"Kunlun failure: api=%s file=%s line=%d status=%d\n",api,file,line,status);check(status);}
}
#define check(...) check_at((__VA_ARGS__),__FILE__,__LINE__,#__VA_ARGS__)
int device_count(int * n) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return xpu_device_count(n);
}
void select(int id) { diagnostic_device=id; require(id >= 0 && id < 2); check(xpu_set_device(id)); }
int wait() { return xpu_wait(); }
int copy(void * dst, const void * src, size_t bytes, CopyKind kind) {
    require(kind == host_to_device || kind == device_to_host || kind == device_to_device);
    auto direction = kind == host_to_device ? XPU_HOST_TO_DEVICE : kind == device_to_host ? XPU_DEVICE_TO_HOST : XPU_DEVICE_TO_DEVICE;
    return xpu_memcpy(dst, src, bytes, direction);
}
void * allocate(int id, size_t bytes) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    select(id);
    void * ptr = nullptr;
    if (xpu_malloc(&ptr, std::max<size_t>(bytes, 256)) != 0) return nullptr;
    allocated[id] += std::max<size_t>(bytes, 256);
    return ptr;
}
void release(int id, void * ptr, size_t bytes) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (ptr) { select(id); check(xpu_free(ptr)); allocated[id] -= std::max<size_t>(bytes, 256); }
}
size_t allocated_bytes(int id) { require(id >= 0 && id < 2); return allocated[id].load(); }
void peer(int dst, void * dp, int src, const void * sp, size_t bytes) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    select(src); check(xpu_wait()); select(dst);
    check(xpu_memcpy_peer(dst, dp, src, const_cast<void *>(sp), bytes));
    check(xpu_wait()); ++peer_copies; peer_bytes += bytes;
}
Buffer::Buffer(int id, size_t bytes): device_(id), bytes_(bytes), ptr_(allocate(id, bytes)) { require(ptr_); }
Buffer::~Buffer() { release(device_, ptr_, bytes_); }
size_t aligned(size_t n) { require(n <= SIZE_MAX - 255); return (n + 255) & ~size_t(255); }
size_t packed_i8_size(int k, int n) { require(k > 0 && n >= 0); return aligned((size_t)k*n) + aligned((size_t)n*4) + 256; }
float * packed_i8_scales(void * p, int k, int n) { return (float *)((char *)p + aligned((size_t)k*n)); }
float * packed_i8_maxima(void * p, int k, int n) { return (float *)((char *)packed_i8_scales(p,k,n) + aligned((size_t)n*4)); }
void upload_i8_rows(int id, void * weights, int k, int n, int first, int count, const float * input) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    require(weights && input && k > 0 && n > 0 && first >= 0 && count >= 0 && first <= n && count <= n-first);
    std::vector<int8_t> quant((size_t)count*k);
    std::vector<float> scales(count);
    for (int r = 0; r < count; ++r) {
        float maximum = 0;
        for (int j=0;j<k;++j) { float v=input[(size_t)r*k+j]; require(std::isfinite(v)); maximum=std::max(maximum,std::abs(v)); }
        scales[r]=maximum > 0 ? maximum/127 : 1;
        float inv=maximum > 0 ? 127/maximum : 0;
        for (int j=0;j<k;++j) {
            float value=input[(size_t)r*k+j]*inv;
            quant[(size_t)r*k+j]=(int8_t)std::max(-127,std::min(127,(int)(value+(value >= 0 ? .5f : -.5f))));
        }
    }
    float maximum[6]={127,127,127,127,127,127};
    select(id);
    check(xpu_memcpy((char *)weights+(size_t)first*k,quant.data(),quant.size(),XPU_HOST_TO_DEVICE));
    check(xpu_memcpy(packed_i8_scales(weights,k,n)+first,scales.data(),scales.size()*4,XPU_HOST_TO_DEVICE));
    check(xpu_memcpy(packed_i8_maxima(weights,k,n),maximum,sizeof(maximum),XPU_HOST_TO_DEVICE));
    check(xpu_wait());
}
struct Kernel::Impl {
    Impl * elementwise = nullptr;
    Impl * broadcast = nullptr;
    Impl * copy_rows = nullptr;
    Impl * set_rows = nullptr;
    Impl * rms_partial = nullptr;
    Impl * rms_output = nullptr;
    Impl * combine = nullptr;
    void * scratch = nullptr;
    size_t scratch_size = 0;
    int id = -1;
    void * alias_input = nullptr;
    size_t alias_input_size = 0;
    void * code = nullptr;
    XPUFunc fn = nullptr;
    std::string symbol;
    xdnn::Context * vendor = nullptr;
    void * quant = nullptr, * correction = nullptr, * input_max = nullptr;
    size_t quant_size = 0, correction_size = 0;
};
void load_kernel(Kernel::Impl & k, const char * name, int id, bool sd = false) {
    if (k.fn) return;
    select(id); k.id = id;
    std::string path = kernel_directory() + "/" + name;
    std::ifstream bin(path + ".bin", std::ios::binary), entry(path + ".entry");
    std::vector<char> bytes((std::istreambuf_iterator<char>(bin)), std::istreambuf_iterator<char>());
    std::getline(entry, k.symbol);
    require(!bytes.empty() && !k.symbol.empty());
    // Keep the final return away from the instruction fetch boundary.
    bytes.resize(((bytes.size() + 255) & ~size_t(255)) + 256, 0);
    require(bytes.size() <= 16384);
    check(xpu_malloc(&k.code, bytes.size()));
    check(xpu_memcpy(k.code, bytes.data(), bytes.size(), XPU_HOST_TO_DEVICE)); check(xpu_wait());
    if (sd) check(xpu_create_sd_func(&k.fn, (uint64_t)(uintptr_t)k.code, bytes.size(), 0, 0, 0, k.symbol.c_str(), true));
    else check(xpu_create_cl_func(&k.fn, (uint64_t)(uintptr_t)k.code, bytes.size(), 0, 0, 0, k.symbol.c_str(), true));
}
void combine_mv(Kernel::Impl & parent, int id, void * y, const void * products,
                 const float * scales, int n, int batch) {
    auto & fn=*parent.combine;
    select(id);
    const void * maximum0=parent.input_max;
    const void * maximum1=(char *)parent.input_max+256;
    size_t offset=0;
    for (const void * arg : {(const void *)&y,(const void *)&products,(const void *)&scales,(const void *)&maximum0,(const void *)&maximum1}) {
        check(xpu_launch_argument_set(arg,8,offset)); offset+=8;
    }
    for (const int * arg : {&n,&batch}) { check(xpu_launch_argument_set(arg,4,offset)); offset+=4; }
    check(xpu_launch_config(4,8)); check(xpu_launch_async(fn.fn));
}

void launch_mv(Kernel::Impl & fn, int id, void * y, const void * w, const void * x, int k, int n, int batch, int type, const float * row_scales, const float * weight_max, int tile) {
    select(id); fn.id = id;
    if (type == 3) {
        const char * mode=std::getenv("KUNLUN_GEMV");
        bool packed=(!mode || std::strcmp(mode,"auto") == 0) ? batch <= 8 : std::strcmp(mode,"fused") == 0;
        bool fused=packed && n <= INT_MAX/8/batch;
        if (fused) {
            if (!fn.combine) fn.combine=new Kernel::Impl{};
            load_kernel(*fn.combine,"kunlun-combine",id);
        }
        if (!fn.vendor) { fn.vendor=xdnn::create_context(); require(fn.vendor); fn.vendor->set_ncluster(4); }
        {
            size_t count = (size_t)k*batch, stride = aligned(count), qs = 2*stride, ys = (size_t)n*batch*4;
            size_t cs=fused ? 2*ys : ys, correction_offset=fused ? count : stride;
            if (fn.quant_size < qs) { release(id,fn.quant,fn.quant_size); fn.quant=allocate(id,qs); fn.quant_size=qs; }
            if (fn.correction_size < cs) { release(id,fn.correction,fn.correction_size); fn.correction=allocate(id,cs); fn.correction_size=cs; }
            if (!fn.input_max) {
                fn.input_max=allocate(id,1024); require(fn.input_max);
                float raw[6]={127,127,127,127,127,127};
                check(xpu_memcpy((char *)fn.input_max+512,raw,sizeof(raw),XPU_HOST_TO_DEVICE)); check(xpu_wait());
            }
            require(fn.quant && fn.correction && fn.input_max);
            std::vector<float> input(count), residual(count);
            std::vector<int8_t> quant(qs);
            check(xpu_memcpy(input.data(),x,count*4,XPU_DEVICE_TO_HOST)); check(xpu_wait());
            float max0=0,max1=0;
            for (float value : input) { require(std::isfinite(value)); max0=std::max(max0,std::abs(value)); }
            if (max0 == 0) max0=1;
            for (size_t j=0;j<count;++j) {
                float value=input[j]*(127/max0);
                quant[j]=(int8_t)std::max(-127,std::min(127,(int)(value+(value >= 0 ? .5f : -.5f))));
                residual[j]=input[j]-quant[j]*(max0/127);
                max1=std::max(max1,std::abs(residual[j]));
            }
            if (max1 == 0) max1=1;
            for (size_t j=0;j<count;++j) {
                float value=residual[j]*(127/max1);
                quant[correction_offset+j]=(int8_t)std::max(-127,std::min(127,(int)(value+(value >= 0 ? .5f : -.5f))));
            }
            float m0[6],m1[6]; std::fill_n(m0,6,max0); std::fill_n(m1,6,max1);
            check(xpu_memcpy(fn.quant,quant.data(),qs,XPU_HOST_TO_DEVICE));
            check(xpu_memcpy(fn.input_max,m0,sizeof(m0),XPU_HOST_TO_DEVICE));
            check(xpu_memcpy((char *)fn.input_max+256,m1,sizeof(m1),XPU_HOST_TO_DEVICE)); check(xpu_wait());
            if (fused) {
                check(xdnn::fc<int8_t,int8_t,float,int8_t>(fn.vendor,(const int8_t *)fn.quant,(const int8_t *)w,(float *)fn.correction,2*batch,n,k,false,true,(const float *)((char *)fn.input_max+512),weight_max,nullptr));
                combine_mv(fn,id,y,fn.correction,row_scales,n,batch);
            } else {
            check(xdnn::fc<int8_t,int8_t,float,int8_t>(fn.vendor,(const int8_t *)fn.quant,(const int8_t *)w,(float *)y,batch,n,k,false,true,(const float *)fn.input_max,weight_max,nullptr));
            check(xdnn::fc<int8_t,int8_t,float,int8_t>(fn.vendor,(const int8_t *)fn.quant+stride,(const int8_t *)w,(float *)fn.correction,batch,n,k,false,true,(const float *)((char *)fn.input_max+256),weight_max,nullptr));
            check(xdnn::add<float>(fn.vendor,(const float *)y,(const float *)fn.correction,(float *)y,n*batch));
            }
        }
        if (!fused) check(xdnn::broadcast_mul<float>(fn.vendor,(const float *)y,row_scales,(float *)y,{batch,n},{1,n}));
        ++matmuls[id]; return;
    }
    load_kernel(fn, "kunlun-matvec", id);
    size_t offset = 0;
    for (const void * arg : { (const void *)&y, (const void *)&w, (const void *)&x }) { check(xpu_launch_argument_set(arg, 8, offset)); offset += 8; }
    for (const int * arg : { &k, &n, &batch, &type, &tile }) { check(xpu_launch_argument_set(arg, 4, offset)); offset += 4; }
    check(xpu_launch_config(4, 8)); check(xpu_launch_async(fn.fn)); ++matmuls[id];
}

Kernel::Kernel(): impl_(new Impl{}) {}
Kernel::~Kernel() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    for (auto * state : {impl_->elementwise,impl_->broadcast,impl_->copy_rows,impl_->set_rows,impl_->rms_partial,impl_->rms_output,impl_->combine,impl_}) {
        if (!state) continue;
        auto & p=*state;
        if (p.id >= 0) {
            select(p.id); check(xpu_wait());
            if (p.vendor) xdnn::destroy_context(p.vendor);
            release(p.id,p.quant,p.quant_size);
            release(p.id,p.correction,p.correction_size);
            release(p.id,p.input_max,1024);
            release(p.id,p.alias_input,p.alias_input_size);
            release(p.id,p.scratch,p.scratch_size);
            if (p.code) check(xpu_free(p.code));
        }
        delete state;
    }
}
void matmul(Kernel & kernel, int id, void * y, const void * w, const void * x,
            int k, int n, int batch, WeightType type, const float * scales, const float * maximum, int tile) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    require(type == WeightType::f32 || type == WeightType::f16 || type == WeightType::q4_0 || type == WeightType::rowwise_i8);
    require(k > 0 && k <= 16384 && k%32 == 0 && n > 0 && batch > 0 && n <= INT_MAX/batch && k <= INT_MAX/batch);
    require((size_t)n*batch <= INT_MAX/4 && (size_t)k*batch <= INT_MAX/4);
    size_t row_bytes=type == WeightType::f32 ? (size_t)k*4 : type == WeightType::f16 ? (size_t)k*2 : type == WeightType::q4_0 ? (size_t)k/32*18 : (size_t)k;
    require(y && w && x && (size_t)n <= INT_MAX/row_bytes && (tile == 1 || tile == 16));
    require(kernel.impl_->id == -1 || kernel.impl_->id == id);
    require(type != WeightType::rowwise_i8 || (scales && maximum));
    launch_mv(*kernel.impl_,id,y,w,x,k,n,batch,(int)type,scales,maximum,tile);
}
void elementwise(Kernel & kernel, int id, void * y, const void * a, const void * b,
                 int n, int width, int bn, ElementwiseOp operation, float parameter) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    int op=(int)operation;
    require(y && a && b && n > 0 && n <= INT_MAX/4 && width > 0 && width <= 16384 && n%width == 0);
    require(op >= 0 && op <= 4 && (bn == n || bn == width));
    if (op == 3) require(std::isfinite(parameter) && parameter > 0);
    require(kernel.impl_->id == -1 || kernel.impl_->id == id);
    kernel.impl_->id=id;
    const char * rms_mode=std::getenv("KUNLUN_RMS_NORM");
    if (op == 3 && width >= 256 && width <= 16384 && width%32 == 0 && n%width == 0 && n/width <= 32 && std::isfinite(parameter) && parameter > 0 &&
        !(rms_mode && std::strcmp(rms_mode,"legacy") == 0) &&
        ((width >= 512 && n/width <= 16) || (rms_mode && std::strcmp(rms_mode,"parallel") == 0))) {
        rms_norm(kernel,id,y,a,width,n/width,parameter); return;
    }
    const char * broadcast_mode=std::getenv("KUNLUN_BROADCAST_VECTOR");
    bool vector_broadcast=(op == 0 || op == 1) && bn != n && bn%256 == 0 && broadcast_mode && std::strcmp(broadcast_mode,"1") == 0;
    auto * & program=vector_broadcast ? kernel.impl_->broadcast : kernel.impl_->elementwise;
    if (!program) program=new Kernel::Impl{};
    auto & fn=*program;
    select(id); load_kernel(fn,vector_broadcast ? "kunlun-elementwise-broadcast" : "kunlun-elementwise",id);
    if (y == a || y == b) {
        size_t bytes=(size_t)(y == a ? n : bn)*4;
        if (fn.alias_input_size < bytes) {
            release(id,fn.alias_input,fn.alias_input_size);
            fn.alias_input=allocate(id,bytes); fn.alias_input_size=bytes;
        }
        require(fn.alias_input);
        const char * mode=std::getenv("KUNLUN_ALIAS_COPY");
        if (mode && std::string(mode) == "legacy") {
            check(xpu_memcpy(fn.alias_input,y,bytes,XPU_DEVICE_TO_DEVICE));
        } else {
            copy_rows(kernel,id,fn.alias_input,y,bytes/4,1,bytes/4,0);
        }
        check(xpu_wait());
        if (b == y) b=fn.alias_input;
        if (a == y) a=fn.alias_input;
    }
    size_t offset=0;
    for (const void * arg : {(const void *)&y,(const void *)&a,(const void *)&b}) { check(xpu_launch_argument_set(arg,8,offset)); offset+=8; }
    for (const int * arg : {&n,&width,&bn,&op}) { check(xpu_launch_argument_set(arg,4,offset)); offset+=4; }
    check(xpu_launch_argument_set(&parameter,4,offset));
    check(xpu_launch_config(4,8)); check(xpu_launch_async(fn.fn)); check(xpu_wait());
}
void copy_rows(Kernel & kernel, int id, void * dst, const void * src,
               int width, int rows, int stride, int column) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    require(dst && src && dst != src && width > 0 && rows > 0 && stride >= width);
    require(column >= 0 && column <= stride-width && rows <= INT_MAX/4/stride);
    uintptr_t d=(uintptr_t)dst, s=(uintptr_t)src;
    size_t db=((size_t)(rows-1)*stride+column+width)*4, sb=(size_t)rows*width*4;
    require(d+db >= d && s+sb >= s && (d+db <= s || s+sb <= d));
    require(kernel.impl_->id == -1 || kernel.impl_->id == id);
    kernel.impl_->id=id;
    if (!kernel.impl_->copy_rows) kernel.impl_->copy_rows=new Kernel::Impl{};
    auto & fn=*kernel.impl_->copy_rows;
    select(id); load_kernel(fn,"kunlun-copy-rows",id);
    size_t offset=0;
    for (const void * arg : {(const void *)&dst,(const void *)&src}) {
        check(xpu_launch_argument_set(arg,8,offset)); offset+=8;
    }
    for (const int * arg : {&width,&rows,&stride,&column}) {
        check(xpu_launch_argument_set(arg,4,offset)); offset+=4;
    }
    check(xpu_launch_config(4,8)); check(xpu_launch_async(fn.fn));
}
bool set_rows_f16(Kernel & kernel, int id, void * dst, size_t dst_bytes,
                  const void * src, size_t src_bytes, const void * indices, size_t index_bytes,
                  const SetRowsLayout & p) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    require(dst && src && indices && dst != src && p.width > 0 && p.rows > 0 && p.dim2 > 0 && p.dim3 > 0);
    require(p.destination_rows > 0 && (p.source_bytes == 2 || p.source_bytes == 4) && (p.index_bytes == 4 || p.index_bytes == 8));
    require(p.index_extent[0] > 0 && p.index_extent[1] > 0 && p.dim2%p.index_extent[0] == 0 && p.dim3%p.index_extent[1] == 0);
    for (int stride : p.source_stride) require(stride > 0);
    for (int stride : p.destination_stride) require(stride > 0);
    for (int stride : p.index_stride) require(stride > 0);
    size_t ss=(size_t)(p.width-1)*p.source_stride[0]+(size_t)(p.rows-1)*p.source_stride[1]+(size_t)(p.dim2-1)*p.source_stride[2]+(size_t)(p.dim3-1)*p.source_stride[3]+p.source_bytes;
    size_t ds=(size_t)p.width*2+(size_t)(p.destination_rows-1)*p.destination_stride[0]+(size_t)(p.dim2-1)*p.destination_stride[1]+(size_t)(p.dim3-1)*p.destination_stride[2];
    size_t is=(size_t)(p.rows-1)*p.index_stride[0]+(size_t)(p.index_extent[0]-1)*p.index_stride[1]+(size_t)(p.index_extent[1]-1)*p.index_stride[2]+p.index_bytes;
    require(ss <= src_bytes && ds <= dst_bytes && is <= index_bytes && ss <= INT_MAX && ds <= INT_MAX && is <= INT_MAX);
    for (int stride : p.source_stride) require(stride%p.source_bytes == 0);
    for (int stride : p.index_stride) require(stride%p.index_bytes == 0);
    for (int stride : p.destination_stride) require(stride%2 == 0);
    size_t row_span=(size_t)(p.destination_rows-1)*p.destination_stride[0]+(size_t)p.width*2;
    size_t head_span=(size_t)(p.dim2-1)*p.destination_stride[1]+row_span;
    require((size_t)p.destination_stride[0] >= (size_t)p.width*2);
    require(p.dim2 == 1 || (size_t)p.destination_stride[1] >= row_span);
    require(p.dim3 == 1 || (size_t)p.destination_stride[2] >= head_span);
    uintptr_t da=(uintptr_t)dst,sa=(uintptr_t)src,ia=(uintptr_t)indices;
    require(da%2 == 0 && sa%p.source_bytes == 0 && ia%p.index_bytes == 0);
    require(da+ds >= da && sa+ss >= sa && ia+is >= ia);
    require((da+ds <= sa || sa+ss <= da) && (da+ds <= ia || ia+is <= da));
    if (p.cache_features) {
        require(p.width == 1 && p.dim2 == 1 && p.dim3 == 1 && p.cache_features > 0 && p.cache_features%8 == 0);
        require(p.cache_tokens > 0 && p.cache_tokens <= 32 && p.cache_stride > 0);
        require((int64_t)p.cache_features*p.cache_tokens == p.rows && (int64_t)p.cache_features*p.cache_stride == p.destination_rows);
        require(p.source_stride[1] == p.source_bytes && p.index_stride[0] == p.index_bytes && p.destination_stride[0] == 2);
    } else require(p.cache_tokens == 0 && p.cache_stride == 0);
    require(kernel.impl_->id == -1 || kernel.impl_->id == id); kernel.impl_->id=id;
    if (!kernel.impl_->set_rows) kernel.impl_->set_rows=new Kernel::Impl{};
    auto & fn=*kernel.impl_->set_rows;
    select(id); load_kernel(fn,"kunlun-set-rows",id);
    if (!fn.scratch) { fn.scratch=allocate(id,512); fn.scratch_size=512; }
    require(fn.scratch);
    static_assert(sizeof(SetRowsLayout)==22*4,"SetRowsLayout layout");
    int config[128]={}; std::memcpy(config,&p,sizeof(p));
    check(xpu_memcpy(fn.scratch,config,sizeof(config),XPU_HOST_TO_DEVICE)); check(xpu_wait());
    void * descriptor=fn.scratch;
    void * status=(char *)fn.scratch+256;
    size_t offset=0;
    for (const void * arg : {(const void *)&dst,(const void *)&src,(const void *)&indices,(const void *)&descriptor,(const void *)&status}) {
        check(xpu_launch_argument_set(arg,8,offset)); offset+=8;
    }
    check(xpu_launch_config(4,8)); check(xpu_launch_async(fn.fn)); check(xpu_wait());
    int invalid=0;
    check(xpu_memcpy(&invalid,status,4,XPU_DEVICE_TO_HOST)); check(xpu_wait());
    return invalid == 0;
}

void rms_norm(Kernel & kernel, int id, void * y, const void * input, int width, int rows, float epsilon) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    require(y && input && width >= 256 && width <= 16384 && width%32 == 0 && rows > 0 && rows <= 32);
    require(std::isfinite(epsilon) && epsilon > 0);
    require(kernel.impl_->id == -1 || kernel.impl_->id == id); kernel.impl_->id=id;
    if (!kernel.impl_->rms_partial) kernel.impl_->rms_partial=new Kernel::Impl{};
    if (!kernel.impl_->rms_output) kernel.impl_->rms_output=new Kernel::Impl{};
    auto & first=*kernel.impl_->rms_partial; auto & second=*kernel.impl_->rms_output;
    select(id); load_kernel(first,"kunlun-rms-partial",id); load_kernel(second,"kunlun-rms-output",id);
    size_t bytes=(size_t)width*rows*4, ps=(size_t)rows*32*4;
    if (first.scratch_size < ps) { release(id,first.scratch,first.scratch_size); first.scratch=allocate(id,ps); first.scratch_size=ps; }
    require(first.scratch);
    const void * x=input;
    if (y == input) {
        if (first.alias_input_size < bytes) { release(id,first.alias_input,first.alias_input_size); first.alias_input=allocate(id,bytes); first.alias_input_size=bytes; }
        require(first.alias_input);
        copy_rows(kernel,id,first.alias_input,input,width*rows,1,width*rows,0); check(xpu_wait()); x=first.alias_input;
    }
    void * partials=first.scratch;
    size_t offset=0;
    for (const void * arg : {(const void *)&x,(const void *)&partials}) { check(xpu_launch_argument_set(arg,8,offset)); offset+=8; }
    for (const int * arg : {&width,&rows}) { check(xpu_launch_argument_set(arg,4,offset)); offset+=4; }
    check(xpu_launch_config(4,8)); check(xpu_launch_async(first.fn)); check(xpu_wait());
    offset=0;
    for (const void * arg : {(const void *)&y,(const void *)&x,(const void *)&partials}) { check(xpu_launch_argument_set(arg,8,offset)); offset+=8; }
    for (const int * arg : {&width,&rows}) { check(xpu_launch_argument_set(arg,4,offset)); offset+=4; }
    check(xpu_launch_argument_set(&epsilon,4,offset));
    check(xpu_launch_config(4,8)); check(xpu_launch_async(second.fn)); check(xpu_wait());
}

namespace {
struct NativeL3 {
    int id;
    void * pointer = nullptr;
    size_t bytes = 0;
    explicit NativeL3(int device): id(device) {}
    ~NativeL3() {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        if (pointer) { select(id); check(xpu_wait()); check(xpu_free(pointer)); }
    }
    bool reserve(size_t size) {
        if (bytes >= size) return true;
        select(id); check(xpu_wait());
        if (pointer) { check(xpu_free(pointer)); pointer=nullptr; bytes=0; }
        int status=xpu_malloc(&pointer,size,XPU_MEM_L3);
        if (status) {
            pointer=nullptr;
            if (status != XPUERR_NOMEM && status != -XPUERR_NOMEM && status != XPUERR_NOSUPPORT && status != -XPUERR_NOSUPPORT) check(status);
            return false;
        }
        require(pointer && ((uintptr_t)pointer&255) == 0); bytes=size; return true;
    }
};
std::weak_ptr<NativeL3> native_l3[2];
struct NativeArgument { const void * value; size_t size; };
template<class T> NativeArgument native_arg(const T & value) { return {&value,sizeof(T)}; }
void native_launch(Kernel::Impl & program, std::initializer_list<NativeArgument> args, int cores) {
    size_t offset=0;
    for (auto arg : args) { check(xpu_launch_argument_set(arg.value,arg.size,offset)); offset+=arg.size; }
    check(xpu_launch_config(4,cores)); check(xpu_launch_async(program.fn));
}
void native_reserve(std::unique_ptr<Buffer> & buffer, int id, size_t size) {
    if (!buffer || buffer->size() < size) { check(xpu_wait()); buffer.reset(new Buffer(id,size)); }
}
}
struct NativeQ4Workspace::Impl {
    int id = -1;
    Kernel::Impl maximum, main, residual, compute, compute_n32, sum, fallback;
    std::unique_ptr<Buffer> quant, rest, peaks0, peaks1, parameters, hbm_partial;
    std::shared_ptr<NativeL3> l3;
    size_t accelerated = 0, fallbacks = 0;
    bool l3_unavailable = false;
};
NativeQ4Workspace::NativeQ4Workspace(): impl_(new Impl{}) {}
NativeQ4Workspace::~NativeQ4Workspace() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto & s=*impl_;
    if (s.id >= 0) {
        select(s.id); check(xpu_wait());
        for (auto * program : {&s.maximum,&s.main,&s.residual,&s.compute,&s.compute_n32,&s.sum,&s.fallback}) {
            if (program->code) check(xpu_free(program->code));
        }
    }
    delete impl_;
}
NativeQ4Statistics NativeQ4Workspace::statistics() const {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto & s=*impl_; size_t hbm=0;
    for (auto * b : {s.quant.get(),s.rest.get(),s.peaks0.get(),s.peaks1.get(),s.parameters.get(),s.hbm_partial.get()})
        if (b) hbm+=b->size();
    return {s.accelerated,s.fallbacks,hbm,s.l3 ? s.l3->bytes : 0};
}
const char * native_q4_reason(int k, int n, int batch, InputType type) {
    if (!native_q4_layout_supported(k,n)) return "K/N/layout capacity";
    if (type != InputType::f32 && type != InputType::f16) return "input type";
    if (batch != 1) return "batch != 1";
    if (n%128) return "rows not divisible by 128 (N64 requires 256)";
    if ((size_t)n*(k/32)*4 > 16*1024*1024) return "partial capacity";
    return "";
}
bool native_q4_matmul(NativeQ4Workspace & workspace, int id, void * y, const void * weights,
                      const void * input, int k, int n, int batch, InputType type) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    require(id >= 0 && id < 2 && native_q4_layout_supported(k,n));
    require(type == InputType::f32 || type == InputType::f16);
    require(batch > 0 && (size_t)n*batch <= INT_MAX/4 && (size_t)k*batch <= INT_MAX/4);
    require(y && weights && input);
    uintptr_t yp=(uintptr_t)y, wp=(uintptr_t)weights, xp=(uintptr_t)input;
    size_t ys=(size_t)n*batch*4, ws=native_q4_size(k,n), xs=(size_t)k*batch*(type == InputType::f16 ? 2 : 4);
    require(yp%4 == 0 && wp%256 == 0 && xp%(type == InputType::f16 ? 2 : 4) == 0);
    require(yp+ys >= yp && wp+ws >= wp && xp+xs >= xp);
    require((yp+ys <= wp || wp+ws <= yp) && (yp+ys <= xp || xp+xs <= yp) && (wp+ws <= xp || xp+xs <= wp));
    auto & s=*workspace.impl_; require(s.id == -1 || s.id == id); s.id=id;
    select(id);
    size_t payload=aligned((((size_t)n+31)&~size_t(31))*k/2);
    const void * scales=(const char *)weights+payload;
    int dtype=type == InputType::f16 ? 1 : 0;
    if (*native_q4_reason(k,n,batch,type) || (yp|xp)&63) {
        load_kernel(s.fallback,"kunlun-native-q4-fallback",id);
        native_launch(s.fallback,{native_arg(y),native_arg(weights),native_arg(scales),native_arg(input),
            native_arg(k),native_arg(n),native_arg(batch),native_arg(dtype)},8);
        check(xpu_wait()); ++s.fallbacks; ++matmuls[id]; return false;
    }
    // All five launches finish before releasing runtime_mutex. SDK L3 allocations
    // own disjoint ranges; shared native scratch is never live across another call.
    load_kernel(s.maximum,"kunlun-native-q4-max",id);
    load_kernel(s.main,"kunlun-native-q4-main",id);
    load_kernel(s.residual,"kunlun-native-q4-residual",id);
    auto & compute=n%256 ? s.compute_n32 : s.compute;
    load_kernel(compute,n%256 ? "kunlun-native-q4-n32" : "kunlun-native-q4",id,true);
    load_kernel(s.sum,"kunlun-native-q4-sum",id);
    size_t quant_size=(size_t)4*k+1024, partial_size=(size_t)n*(k/32)*4;
    if (!s.quant || s.quant->size() < quant_size) {
        native_reserve(s.quant,id,quant_size);
        std::vector<char> zero(quant_size,0);
        check(xpu_memcpy(s.quant->data(),zero.data(),zero.size(),XPU_HOST_TO_DEVICE)); check(xpu_wait());
    }
    native_reserve(s.rest,id,(size_t)k*4);
    native_reserve(s.peaks0,id,2048); native_reserve(s.peaks1,id,2048); native_reserve(s.parameters,id,64);
    if (!s.l3) {
        s.l3=native_l3[id].lock();
        if (!s.l3) { s.l3=std::make_shared<NativeL3>(id); native_l3[id]=s.l3; }
    }
    void * partial=nullptr;
    if (!s.l3_unavailable && s.l3->reserve(partial_size)) partial=s.l3->pointer;
    else {
        if (!s.l3_unavailable) std::fprintf(stderr,"Kunlun native Q4 PD%d: L3 unavailable; using HBM partials (%zu bytes)\n",id,partial_size);
        s.l3_unavailable=true;
        native_reserve(s.hbm_partial,id,partial_size); partial=s.hbm_partial->data();
    }
    void * q=s.quant->data(), * rest=s.rest->data(), * p0=s.peaks0->data(), * p1=s.peaks1->data(), * params=s.parameters->data();
    native_launch(s.maximum,{native_arg(p0),native_arg(input),native_arg(k),native_arg(dtype)},16);
    native_launch(s.main,{native_arg(q),native_arg(rest),native_arg(p1),native_arg(input),native_arg(p0),native_arg(k),native_arg(dtype)},16);
    native_launch(s.residual,{native_arg(q),native_arg(params),native_arg(rest),native_arg(p0),native_arg(p1),native_arg(k)},16);
    check(xpu_wait());
    int first=0, components=2, bits=8, pipelined=1, fold=0, mode=1; float a0=1,a1=1,a2=1;
    native_launch(compute,{native_arg(partial),native_arg(weights),native_arg(q),native_arg(k),native_arg(n),
        native_arg(batch),native_arg(first),native_arg(n),native_arg(components),native_arg(bits),native_arg(a0),
        native_arg(a1),native_arg(a2),native_arg(pipelined),native_arg(scales),native_arg(fold),native_arg(params)},8);
    check(xpu_wait());
    native_launch(s.sum,{native_arg(y),native_arg(partial),native_arg(scales),native_arg(k),native_arg(n),
        native_arg(batch),native_arg(first),native_arg(n),native_arg(mode)},16);
    check(xpu_wait()); ++s.accelerated; ++matmuls[id]; return true;
}

#include "q4-common.inc"

Statistics statistics() { return {{matmuls[0].load(),matmuls[1].load()},peer_copies.load(),peer_bytes.load()}; }
const char * version() { return "0.4.0"; }
} // namespace kunlun
