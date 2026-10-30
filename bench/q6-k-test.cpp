#include <kunlun/k200.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

static float from_half(unsigned short bits) {
    _Float16 value;
    std::memcpy(&value, &bits, 2);
    return value;
}
// Decode independently in the original ql/qh lane order used by GGML.
static void decode(const unsigned char *b, float *weights) {
    float d = from_half(b[208] | b[209] << 8);
    for (int half = 0; half < 2; ++half) {
        for (int l = 0; l < 32; ++l) {
            for (int lane = 0; lane < 4; ++lane) {
                int lo = half*64 + l + (lane%2)*32;
                int v = lane < 2 ? b[lo]&15 : b[lo]>>4;
                int q = (v | ((b[128 + half*32 + l] >> (2*lane)) & 3) << 4) - 32;
                int scale = (int8_t)b[192 + half*8 + l/16 + lane*2];
                weights[half*128 + l + lane*32] = (d*scale)*q;
            }
        }
    }
}
int main(int argc, char **argv) {
    using namespace kunlun;
    require(argc == 6);
    int k = std::atoi(argv[3]), n = std::atoi(argv[4]), pd = std::atoi(argv[5]);
    require(q6_k_supported(k,n));
    std::vector<unsigned char> raw((size_t)n*(k/256)*210);
    std::ifstream input(argv[1], std::ios::binary);
    input.seekg(std::strtoull(argv[2], nullptr, 10));
    input.read((char *)raw.data(), raw.size());
    require((size_t)input.gcount() == raw.size());
    std::vector<float> x(k), xa(k), result(n+128, -12345.f);
    for (int j=0;j<k;++j) x[j] = std::sin(j*.123f)*.4f;
    for (int b=0;b<k;b+=256) {
        float mx=0;
        for (int j=0;j<256;++j) mx=std::max(mx,std::abs(x[b+j]));
        float inv=mx?127.f/mx:0, scale=mx/127.f;
        for (int j=0;j<256;++j) {
            float v=x[b+j]*inv;
            xa[b+j]=(int)(v+(v>=0?.5f:-.5f))*scale;
        }
    }
    Q6KWeight weight(pd,k,n);
    weight.upload(raw.data(),raw.size());
    Q6KWorkspace workspace;
    Buffer dx(pd,k*4), dy(pd,result.size()*4);
    select(pd);
    check(copy(dx.data(),x.data(),k*4,host_to_device));
    check(copy(dy.data(),result.data(),result.size()*4,host_to_device));
    auto run=[&]{q6_k_matmul(workspace,pd,(float*)dy.data()+64,weight,dx.data());};
    run();
    check(copy(result.data(),dy.data(),result.size()*4,device_to_host));
    for(int j=0;j<64;++j) require(result[j]==-12345 && result[n+64+j]==-12345);
    double err=0, energy=0, implementation_err=0, implementation_energy=0, maximum=0;
    float decoded[256];
    for(int r=0;r<n;++r) {
        double exact=0, quantized=0;
        for(int b=0;b<k/256;++b) {
            decode(raw.data()+((size_t)r*(k/256)+b)*210,decoded);
            for(int j=0;j<256;++j) {
                exact+=double(decoded[j])*x[b*256+j];
                quantized+=double(decoded[j])*xa[b*256+j];
            }
        }
        float y=result[64+r]; require(std::isfinite(y) && y!=-12345);
        double delta=y-exact;err+=delta*delta;energy+=exact*exact;
        maximum=std::max(maximum,std::abs(delta));
        delta=y-quantized;implementation_err+=delta*delta;
        implementation_energy+=quantized*quantized;
    }
    require(implementation_err/implementation_energy<1e-10 && err/energy<1e-3);
    std::vector<double> elapsed;
    for(int i=0;i<3;++i) {
        auto start=std::chrono::steady_clock::now();run();
        elapsed.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
    }
    std::sort(elapsed.begin(),elapsed.end());
    std::printf("Q6_TEST k=%d n=%d pd=%d median_ms=%.6f exact_nmse=%.12g implementation_nmse=%.12g maxabs=%.12g guard=pass payload=%zu workspace=%zu\n",k,n,pd,elapsed[1],err/energy,implementation_err/implementation_energy,maximum,weight.info().payload_bytes,workspace.statistics().workspace_bytes);
    std::fill(x.begin(),x.end(),0);
    check(copy(dx.data(),x.data(),k*4,host_to_device));run();
    check(copy(result.data(),dy.data(),result.size()*4,device_to_host));
    for(int j=0;j<n;++j)require(result[64+j]==0);
    std::puts("Q6_STANDALONE_PASS");
}
