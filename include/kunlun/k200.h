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
    // Optional validated contiguous cache mapping; all zero selects generic indices.
    int cache_features, cache_tokens, cache_stride;
};

class Kernel {
public:
    Kernel();
    ~Kernel();
    Kernel(const Kernel &) = delete;
    Kernel & operator=(const Kernel &) = delete;
    struct Impl;
private:
    Impl * impl_;
    friend void matmul(Kernel &, int, void *, const void *, const void *, int, int, int,
                       WeightType, const float *, const float *, int);
    friend void elementwise(Kernel &, int, void *, const void *, const void *, int, int,
                            int, ElementwiseOp, float);
    friend void copy_rows(Kernel &, int, void *, const void *, int, int, int, int);
    friend bool set_rows_f16(Kernel &, int, void *, size_t, const void *, size_t, const void *, size_t, const SetRowsLayout &);
    friend void rms_norm(Kernel &, int, void *, const void *, int, int, float);
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
// Returns false for an invalid index; valid updates may already have completed.
bool set_rows_f16(Kernel &, int device, void * dst, size_t dst_bytes,
                  const void * src, size_t src_bytes, const void * indices, size_t index_bytes,
                  const SetRowsLayout &);

// FP32 two-stage reduction, width 256..16384 in multiples of 32, up to 32 rows.
// Identical input/output pointers are supported. Completes before returning.
void rms_norm(Kernel &, int device, void * y, const void * x, int width, int rows, float epsilon);

struct Statistics {
    size_t matmuls[2];
    size_t peer_copies, peer_bytes;
};
Statistics statistics();
const char * version();

} // namespace kunlun
