#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>

namespace kunlun {

// The runtime device selection and launch arguments are process global.
std::recursive_mutex & runtime_mutex();
void check(int status);
// Optional borrowed tensor name; dimensions are copied. Clear after graph execution.
void diagnostic_tensor(const char * name, const int64_t * dimensions = nullptr);
void require(bool condition);
int device_count(int * count);
void select(int device);
int wait();
enum CopyKind { host_to_device, device_to_host, device_to_device };
int copy(void * dst, const void * src, size_t bytes, CopyKind kind);
void peer(int dst_device, void * dst, int src_device, const void * src, size_t bytes);
void * allocate(int device, size_t bytes);
void release(int device, void * ptr, size_t bytes);
size_t allocated_bytes(int device);

class Buffer {
public:
    Buffer(int device, size_t bytes);
    ~Buffer();
    Buffer(const Buffer &) = delete;
    Buffer & operator=(const Buffer &) = delete;
    void * data() const { return ptr_; }
    size_t size() const { return bytes_; }
private:
    int device_;
    size_t bytes_;
    void * ptr_;
};

enum class WeightType { f32 = 0, f16 = 1, q4_0 = 2, rowwise_i8 = 3 };
enum class ElementwiseOp { add = 0, multiply = 1, silu = 2, rms_norm = 3, scale = 4 };

struct SetRowsLayout {
    int width, rows, dim2, dim3;
    int source_stride[4];
    int destination_stride[3];
    int index_stride[3];
    int index_extent[2];
    int source_bytes, index_bytes, destination_rows;
    // Optional contiguous cache mapping; all zero selects generic indices.
    // Features are a multiple of 8; 1 <= tokens <= stride (since 0.4.1).
    // Device validation checks index[token*features+f] == f*stride+start+token.
    int cache_features, cache_tokens, cache_stride;
};

struct Flash4Layout;struct FlashStatistics;
struct SwiGLULayout;
struct Matmul4Layout;struct Softmax4Layout;struct Tensor4Layout;struct Rope4Layout;struct AttentionStatistics;

class Kernel {
public:
    Kernel();
    ~Kernel();
    Kernel(const Kernel &) = delete;
    Kernel & operator=(const Kernel &) = delete;
    struct Impl;
private:
    Impl * impl_;
    friend FlashStatistics flash_statistics(const Kernel &);
    friend bool flash_attention(Kernel &,int,void *,size_t,const void *,size_t,const void *,size_t,const void *,size_t,const void *,size_t,const Flash4Layout &);
    friend bool attention_matmul(Kernel &,int,void *,size_t,const void *,size_t,const void *,size_t,const Matmul4Layout &);
    friend bool attention_softmax(Kernel &,int,void *,size_t,const void *,size_t,const void *,size_t,const Softmax4Layout &);
    friend bool attention_contiguous(Kernel &,int,void *,size_t,const void *,size_t,const Tensor4Layout &);
    friend bool attention_rope(Kernel &,int,void *,size_t,const void *,size_t,const int32_t *,size_t,const Rope4Layout &);
    friend AttentionStatistics attention_statistics(const Kernel &);
    friend void matmul(Kernel &, int, void *, const void *, const void *, int, int, int,
                       WeightType, const float *, const float *, int);
    friend void elementwise(Kernel &, int, void *, const void *, const void *, int, int,
                            int, ElementwiseOp, float);
    friend void copy_rows(Kernel &, int, void *, const void *, int, int, int, int);
    friend bool set_rows_f16(Kernel &, int, void *, size_t, const void *, size_t, const void *, size_t, const SetRowsLayout &);
    friend void rms_norm(Kernel &, int, void *, const void *, int, int, float);
    friend bool swiglu(Kernel &, int, void *, size_t, const void *, size_t, const void *, size_t, const SwiGLULayout &);
};

size_t aligned(size_t bytes);
size_t packed_i8_size(int k, int rows);
float * packed_i8_scales(void * weights, int k, int rows);
float * packed_i8_maxima(void * weights, int k, int rows);
// Upload F32 rows to a caller-owned packed_i8_size(k, rows) device allocation.
void upload_i8_rows(int device, void * weights, int k, int rows,
                    int first_row, int count, const float * input);

// W: [n,k], X: [batch,k], Y: [batch,n]. All pointers refer to device memory.
// Q4_0: 18-byte blocks (little-endian F16 scale + 16 paired 4-bit values).
// rowwise_i8 requires scales[n] and maxima[6], also in device memory.
// matmul submits work; select(device) + check(wait()) completes it.
// elementwise completes before returning. K must be divisible by 32, K <= 16384.
void matmul(Kernel &, int device, void * y, const void * w, const void * x,
            int k, int n, int batch, WeightType type,
            const float * scales = nullptr, const float * maxima = nullptr, int tile = 16);
void elementwise(Kernel &, int device, void * y, const void * a, const void * b,
                 int count, int width, int b_count, ElementwiseOp op, float parameter = 0);

// Copy contiguous F32 rows into a strided device output. Pointers must not overlap.
// Stride and offset are in F32 elements. Submits work; wait() completes it.
void copy_rows(Kernel &, int device, void * dst, const void * src,
               int width, int rows, int dst_stride, int dst_offset);

// Indexed F16 updates. Strides are bytes; source/index elements are 2/4 and 4/8 bytes.
// Requires element-aligned strides and non-overlapping destination rows/head regions.
// Source/index storage must not overlap the destination.
// Returns false for an invalid index; valid updates may already have completed.
bool set_rows_f16(Kernel &, int device, void * dst, size_t dst_bytes,
                  const void * src, size_t src_bytes, const void * indices, size_t index_bytes,
                  const SetRowsLayout &);

// FP32 two-stage reduction, width 256..16384 in multiples of 32, up to 32 rows.
// Requires finite positive epsilon. Identical input/output pointers are supported.
// Completes before returning.
void rms_norm(Kernel &, int device, void * y, const void * x, int width, int rows, float epsilon);

// Native layout keeps the original Q4_0 values and FP16 block scales. N is padded to 32.
bool native_q4_layout_supported(int k, int n);
size_t native_q4_size(int k, int n);
bool pack_native_q4(void * destination, size_t bytes, const void * q4, int k, int n);
void upload_native_q4(int device, void * weights, const void * q4, int k, int n);
enum class InputType { f32, f16 };
// Byte strides describe a non-overlapping four-dimensional device view.
struct Tensor4Layout { int ne[4],nb[4];InputType type; };
// SDNN uses half inputs, symmetric INT16 DMA conversion and visible F32 accumulation/output.
// KUNLUN_ATTENTION_MATMUL=cl selects the F32 SIMD arithmetic reference; see docs/attention.md.
struct Matmul4Layout { Tensor4Layout a,b,y;bool round_b_to_f16=true; };
struct Softmax4Layout { Tensor4Layout x,y,mask;bool has_mask=false;float scale=1,max_bias=0; };
struct Rope4Layout { Tensor4Layout x,y;int n_dims,mode,n_offs;float freq_base,freq_scale,attn_factor; };
struct AttentionStatistics { size_t matmuls,softmax,copies,rope,hbm_workspace_bytes,l3_workspace_bytes,rope_host_trig,pack_calls; };
const char *tensor4_reason(const Tensor4Layout &);
const char *attention_matmul_reason(const Matmul4Layout &);
const char *attention_softmax_reason(const Softmax4Layout &);
const char *attention_rope_reason(const Rope4Layout &);
// All calls validate contracts before submission, serialize on runtime_mutex and complete before returning.
bool attention_matmul(Kernel &,int device,void *y,size_t y_bytes,const void *a,size_t a_bytes,const void *b,size_t b_bytes,const Matmul4Layout &);
bool attention_softmax(Kernel &,int device,void *y,size_t y_bytes,const void *x,size_t x_bytes,const void *mask,size_t mask_bytes,const Softmax4Layout &);
bool attention_contiguous(Kernel &,int device,void *y,size_t y_bytes,const void *x,size_t x_bytes,const Tensor4Layout &);
bool attention_rope(Kernel &,int device,void *y,size_t y_bytes,const void *x,size_t x_bytes,const int32_t *positions,size_t position_bytes,const Rope4Layout &);
AttentionStatistics attention_statistics(const Kernel &);

struct SwiGLULayout {
    int width, rows;
    int gate_stride, up_stride, output_stride; // Bytes between rows.
    InputType type;
};
// Empty reason means supported. Inner elements must be contiguous F32 or F16.
const char * swiglu_reason(const SwiGLULayout &);
// Computes SiLU(gate)*up with F32 intermediates and one final output rounding.
// Completes before returning. No HBM workspace or host activation transfer.
// Exact input/output alias is supported with equal row strides; partial alias is not.
// False means an unsupported layout or pointer contract; no work was submitted.
bool swiglu(Kernel &, int device, void * y, size_t y_bytes,
            const void * gate, size_t gate_bytes, const void * up, size_t up_bytes,
            const SwiGLULayout &);

// Empty string means the accelerated SD path is supported. The CL fallback accepts
// all valid native layouts, batch >= 1, and F32/F16 input. No CPU activation copy.
const char * native_q4_reason(int k, int n, int batch, InputType type);
struct NativeQ4Statistics {
    size_t accelerated, fallback;
    size_t hbm_workspace_bytes, l3_workspace_bytes;
};
class NativeQ4Workspace {
public:
    NativeQ4Workspace();
    ~NativeQ4Workspace();
    NativeQ4Workspace(const NativeQ4Workspace &) = delete;
    NativeQ4Workspace & operator=(const NativeQ4Workspace &) = delete;
    NativeQ4Statistics statistics() const;
    struct Impl;
private:
    Impl * impl_;
    friend bool native_q4_matmul(NativeQ4Workspace &, int, void *, const void *, const void *, int, int, int, InputType);
};
// Completes before returning under runtime_mutex; the workspace is bound to one PD.
// Returns true for SD, false for exact CL fallback. Device errors remain fail-fast.
// weights/input/output must not overlap. SD requires 64-byte-aligned pointers.
bool native_q4_matmul(NativeQ4Workspace &, int device, void * y, const void * weights,
                      const void * input, int k, int n, int batch = 1, InputType type = InputType::f32);

// Common Q4 ABI v1. Separate type prevents legacy native layout dispatch.
struct CommonQ4Info {
    uint32_t abi_version, quant_type, layout, scale_order;
    int k, n, kp, np, device;
    size_t scale_offset, payload_bytes;
};
bool common_q4_supported(int k, int n);
size_t common_q4_size(int k, int n);
struct CommonQ4WeightStatistics { size_t uploads, upload_bytes; double pack_ms, upload_ms; };
class CommonQ4Weight {
public:
    // Borrowed storage must be 256-byte aligned and outlive this handle and calls.
    // Null storage allocates an owned Buffer. No upload occurs in a forward call.
    CommonQ4Weight(int device, int k, int n, void * storage = nullptr, size_t bytes = 0);
    ~CommonQ4Weight();
    CommonQ4Weight(const CommonQ4Weight &) = delete;
    CommonQ4Weight & operator=(const CommonQ4Weight &) = delete;
    CommonQ4Info info() const;
    CommonQ4WeightStatistics statistics() const;
    void * data() const;
    void upload(const void * q4, size_t bytes);
    struct Impl;
private:
    Impl * impl_;
    friend void common_q4_matmul(class CommonQ4Workspace &, int, void *, const CommonQ4Weight &, const void *, int, InputType);
};
struct CommonQ4Statistics {
    // GEMV, Combined, Wide, Rows64, Rows128, Combo2, Combo4, exact CL.
    size_t paths[8], calls, chunks;
    size_t hbm_workspace_bytes, l3_workspace_bytes, slice_bytes;
};
struct CommonQ4Quantization { const void * values; const void * scales; int rows, kp, components; };
class CommonQ4Workspace {
public:
    CommonQ4Workspace();
    ~CommonQ4Workspace();
    CommonQ4Workspace(const CommonQ4Workspace &) = delete;
    CommonQ4Workspace & operator=(const CommonQ4Workspace &) = delete;
    CommonQ4Statistics statistics() const;
    // Borrowed device buffers, valid until the next call or workspace destruction.
    CommonQ4Quantization quantization() const;
    struct Impl;
private:
    Impl * impl_;
    friend void common_q4_matmul(CommonQ4Workspace &, int, void *, const CommonQ4Weight &, const void *, int, InputType);
};
// Contiguous X[M,K], Y[M,N]. Synchronous under runtime_mutex, one workspace/PD.
// M>1 uses Row A8, M=1 uses two-component A8. Original weight scales are exact.
// Invalid type, PD ownership, overlap or element alignment is rejected before launch.
void common_q4_matmul(CommonQ4Workspace &, int device, void * y, const CommonQ4Weight &,
                       const void * input, int m, InputType type = InputType::f32);

struct Statistics {
    size_t matmuls[2];
    size_t peer_copies, peer_bytes;
};
Statistics statistics();
const char * version();

} // namespace kunlun

namespace kunlun {
// Experimental opt-in fused Attention; Q/K/V bases must be 32-byte aligned.
struct FlashStatistics { size_t calls,hbm_workspace_bytes,l3_workspace_bytes; int block,query_tile; };
FlashStatistics flash_statistics(const Kernel &);
struct Flash4Layout { Tensor4Layout q,k,v,y,mask; bool has_mask=false; float scale=1,max_bias=0; };
const char *flash_attention_reason(const Flash4Layout &);
bool flash_attention(Kernel &,int,void *,size_t,const void *,size_t,const void *,size_t,const void *,size_t,const void *,size_t,const Flash4Layout &);
}
