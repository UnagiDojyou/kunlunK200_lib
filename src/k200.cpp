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

namespace kunlun {
namespace xdnn = baidu::xpu::api;
namespace {
std::recursive_mutex mutex;
std::atomic<size_t> allocated[2];
std::atomic<size_t> matmuls[2], peer_copies, peer_bytes;
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
void check(int rc) {
    if (rc) { std::fprintf(stderr, "Kunlun runtime error %d\n", rc); std::abort(); }
}
void require(bool ok) {
    if (!ok) { std::fprintf(stderr, "Kunlun invalid argument or unavailable resource\n"); std::abort(); }
}
int device_count(int * n) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return xpu_device_count(n);
}
void select(int id) { require(id >= 0 && id < 2); check(xpu_set_device(id)); }
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
    Impl * copy_rows = nullptr;
    Impl * set_rows = nullptr;
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
void load_kernel(Kernel::Impl & k, const char * name, int id) {
    if (k.fn) return;
    select(id); k.id = id;
    std::string path = kernel_directory() + "/" + name;
    std::ifstream bin(path + ".bin", std::ios::binary), entry(path + ".entry");
    std::vector<char> bytes((std::istreambuf_iterator<char>(bin)), std::istreambuf_iterator<char>());
    std::getline(entry, k.symbol);
    require(!bytes.empty() && !k.symbol.empty());
    // Keep the final return away from the instruction fetch boundary.
    bytes.resize(((bytes.size() + 255) & ~size_t(255)) + 256, 0);
    check(xpu_malloc(&k.code, bytes.size()));
    check(xpu_memcpy(k.code, bytes.data(), bytes.size(), XPU_HOST_TO_DEVICE)); check(xpu_wait());
    // These kernels run on cluster SIMD cores, not the SD/CDNN engines.
    check(xpu_create_cl_func(&k.fn, (uint64_t)(uintptr_t)k.code, bytes.size(), 0, 0, 0, k.symbol.c_str(), true));
}
void launch_mv(Kernel::Impl & fn, int id, void * y, const void * w, const void * x, int k, int n, int batch, int type, const float * row_scales, const float * weight_max, int tile) {
    select(id); fn.id = id;
    if (type == 3) {
        if (!fn.vendor) { fn.vendor=xdnn::create_context(); require(fn.vendor); fn.vendor->set_ncluster(4); }
        {
            size_t count = (size_t)k*batch, stride = aligned(count), qs = 2*stride, ys = (size_t)n*batch*4;
            if (fn.quant_size < qs) { release(id,fn.quant,fn.quant_size); fn.quant=allocate(id,qs); fn.quant_size=qs; }
            if (fn.correction_size < ys) { release(id,fn.correction,fn.correction_size); fn.correction=allocate(id,ys); fn.correction_size=ys; }
            if (!fn.input_max) fn.input_max=allocate(id,512);
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
                quant[stride+j]=(int8_t)std::max(-127,std::min(127,(int)(value+(value >= 0 ? .5f : -.5f))));
            }
            float m0[6],m1[6]; std::fill_n(m0,6,max0); std::fill_n(m1,6,max1);
            check(xpu_memcpy(fn.quant,quant.data(),qs,XPU_HOST_TO_DEVICE));
            check(xpu_memcpy(fn.input_max,m0,sizeof(m0),XPU_HOST_TO_DEVICE));
            check(xpu_memcpy((char *)fn.input_max+256,m1,sizeof(m1),XPU_HOST_TO_DEVICE)); check(xpu_wait());
            check(xdnn::fc<int8_t,int8_t,float,int8_t>(fn.vendor,(const int8_t *)fn.quant,(const int8_t *)w,(float *)y,batch,n,k,false,true,(const float *)fn.input_max,weight_max,nullptr));
            check(xdnn::fc<int8_t,int8_t,float,int8_t>(fn.vendor,(const int8_t *)fn.quant+stride,(const int8_t *)w,(float *)fn.correction,batch,n,k,false,true,(const float *)((char *)fn.input_max+256),weight_max,nullptr));
            check(xdnn::add<float>(fn.vendor,(const float *)y,(const float *)fn.correction,(float *)y,n*batch));
        }
        check(xdnn::broadcast_mul<float>(fn.vendor,(const float *)y,row_scales,(float *)y,{batch,n},{1,n}));
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
    for (auto * state : {impl_->elementwise,impl_->copy_rows,impl_->set_rows,impl_}) {
        if (!state) continue;
        auto & p=*state;
        if (p.id >= 0) {
            select(p.id); check(xpu_wait());
            if (p.vendor) xdnn::destroy_context(p.vendor);
            release(p.id,p.quant,p.quant_size);
            release(p.id,p.correction,p.correction_size);
            release(p.id,p.input_max,512);
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
    require(kernel.impl_->id == -1 || kernel.impl_->id == id);
    kernel.impl_->id=id;
    if (!kernel.impl_->elementwise) kernel.impl_->elementwise=new Kernel::Impl{};
    auto & fn=*kernel.impl_->elementwise;
    select(id); load_kernel(fn,"kunlun-elementwise",id);
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

Statistics statistics() { return {{matmuls[0].load(),matmuls[1].load()},peer_copies.load(),peer_bytes.load()}; }
const char * version() { return "0.1.0"; }
} // namespace kunlun
