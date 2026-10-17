#include <kunlun/k200.h>
#include <xpu/runtime.h>
#include <xpu/xdnn.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
    auto & p=*impl_;
    if (p.id >= 0) {
        select(p.id); check(xpu_wait());
        if (p.vendor) xdnn::destroy_context(p.vendor);
        release(p.id,p.quant,p.quant_size);
        release(p.id,p.correction,p.correction_size);
        release(p.id,p.input_max,512);
        release(p.id,p.alias_input,p.alias_input_size);
        if (p.code) check(xpu_free(p.code));
    }
    delete impl_;
}
void matmul(Kernel & kernel, int id, void * y, const void * w, const void * x,
            int k, int n, int batch, WeightType type, const float * scales, const float * maximum, int tile) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
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
    auto & fn=*kernel.impl_;
    select(id); load_kernel(fn,"kunlun-elementwise",id);
    if (op == 3 && y == a) {
        size_t bytes=(size_t)n*4;
        if (fn.alias_input_size < bytes) {
            release(id,fn.alias_input,fn.alias_input_size);
            fn.alias_input=allocate(id,bytes); fn.alias_input_size=bytes;
        }
        require(fn.alias_input);
        check(xpu_memcpy(fn.alias_input,a,bytes,XPU_DEVICE_TO_DEVICE)); check(xpu_wait());
        a=fn.alias_input;
    }
    size_t offset=0;
    for (const void * arg : {(const void *)&y,(const void *)&a,(const void *)&b}) { check(xpu_launch_argument_set(arg,8,offset)); offset+=8; }
    for (const int * arg : {&n,&width,&bn,&op}) { check(xpu_launch_argument_set(arg,4,offset)); offset+=4; }
    check(xpu_launch_argument_set(&parameter,4,offset));
    check(xpu_launch_config(4,8)); check(xpu_launch_async(fn.fn)); check(xpu_wait());
}
Statistics statistics() { return {{matmuls[0].load(),matmuls[1].load()},peer_copies.load(),peer_bytes.load()}; }
const char * version() { return "0.1.0"; }
} // namespace kunlun
