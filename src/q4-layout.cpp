#include <kunlun/k200.h>
#include <algorithm>
#include <cstring>
#include <vector>

namespace kunlun {
bool native_q4_layout_supported(int k, int n) {
    return k > 0 && k <= 16384 && k%32 == 0 && n > 0 && n <= 90000000/k;
}
size_t native_q4_size(int k, int n) {
    if (!native_q4_layout_supported(k,n)) return 0;
    size_t rows=((size_t)n+31)&~size_t(31);
    return aligned(rows*k/2)+aligned(rows*(k/32)*2);
}
bool pack_native_q4(void * destination, size_t bytes, const void * source, int k, int n) {
    size_t need=native_q4_size(k,n);
    if (!need || bytes < need || !destination || !source) return false;
    uintptr_t dst=(uintptr_t)destination, src=(uintptr_t)source;
    size_t source_bytes=(size_t)n*(k/32)*18;
    if (dst+need < dst || src+source_bytes < src || !(dst+need <= src || src+source_bytes <= dst)) return false;
    auto * out=static_cast<unsigned char *>(destination);
    auto * input=static_cast<const unsigned char *>(source);
    size_t payload=aligned((((size_t)n+31)&~size_t(31))*k/2);
    std::memset(out,0,need);
    // [N32 tile, K32 block, paired K byte, N lane]. Only nibble order changes.
    for (int row=0;row<n;++row) for (int b=0;b<k/32;++b) {
        const auto * block=input+((size_t)row*(k/32)+b)*18;
        size_t tile_block=(size_t)(row/32)*(k/32)+b;
        std::memcpy(out+payload+(tile_block*32+row%32)*2,block,2);
        for (int j=0;j<16;++j) {
            int lo=((block[2+(2*j)%16]>>((2*j)/16*4))&15)-8;
            int hi=((block[2+(2*j+1)%16]>>((2*j+1)/16*4))&15)-8;
            out[tile_block*512+j*32+row%32]=(unsigned char)((lo&15)|((hi&15)<<4));
        }
    }
    return true;
}
void upload_native_q4(int device, void * weights, const void * q4, int k, int n) {
    std::lock_guard<std::recursive_mutex> lock(runtime_mutex());
    std::vector<unsigned char> packed(native_q4_size(k,n));
    require(pack_native_q4(packed.data(),packed.size(),q4,k,n));
    select(device);
    check(copy(weights,packed.data(),packed.size(),host_to_device));
    check(wait());
}
} // namespace kunlun
