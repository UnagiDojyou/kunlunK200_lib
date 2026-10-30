#include <kunlun/k200.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
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
    double err=0, energy=0, implementation_err=0, implementation_energy=0, maximum=0, implementation_maximum=0;
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
        implementation_maximum=std::max(implementation_maximum,std::abs(delta));
        implementation_energy+=quantized*quantized;
    }
    require(implementation_err/implementation_energy<1e-10 && err/energy<1e-3);
    const char *setting=std::getenv("KUNLUN_Q6_K_LEGACY");
    bool was_set=setting;
    std::string saved=setting?setting:"";
    std::vector<float> first=result, other(result.size());
    setenv("KUNLUN_Q6_K_LEGACY",saved=="1"?"0":"1",1);
    run();check(copy(other.data(),dy.data(),other.size()*4,device_to_host));
    double cross_error=0,cross_energy=0,cross_maximum=0;
    for(int j=0;j<n;++j){
        require(std::isfinite(other[64+j]));
        double delta=double(other[64+j])-first[64+j];
        cross_error+=delta*delta;cross_energy+=double(first[64+j])*first[64+j];
        cross_maximum=std::max(cross_maximum,std::abs(delta));
    }
    require(cross_error/cross_energy<1e-10);
    if(was_set)setenv("KUNLUN_Q6_K_LEGACY",saved.c_str(),1);else unsetenv("KUNLUN_Q6_K_LEGACY");
    run();check(copy(other.data(),dy.data(),other.size()*4,device_to_host));
    require(std::memcmp(first.data(),other.data(),first.size()*4)==0);
    std::printf("Q6_MODE_REUSE cross_nmse=%.12g cross_maxabs=%.12g a8_f64_maxabs=%.12g repeat_bitwise=1\n",cross_error/cross_energy,cross_maximum,implementation_maximum);
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
    // Distinct chunk magnitudes expose an A8 scale reused across chunk256.
    for(int j=0;j<k;++j)x[j]=float(j%17-8)*std::ldexp(1.f,(j/256)%11-5);
    std::fill(x.begin(),x.begin()+256,0.f);
    for(int b=0;b<k;b+=256){
        float mx=0;for(int j=0;j<256;++j)mx=std::max(mx,std::abs(x[b+j]));
        float inv=mx?127.f/mx:0,scale=mx/127.f;
        for(int j=0;j<256;++j){float v=x[b+j]*inv;xa[b+j]=int(v+(v>=0?.5f:-.5f))*scale;}
    }
    check(copy(dx.data(),x.data(),k*4,host_to_device));run();
    check(copy(result.data(),dy.data(),result.size()*4,device_to_host));
    double chunk_error=0,chunk_energy=0,chunk_maximum=0;
    for(int r=0;r<n;++r){
        double ref=0;
        for(int b=0;b<k/256;++b){
            decode(raw.data()+((size_t)r*(k/256)+b)*210,decoded);
            for(int j=0;j<256;++j)ref+=double(decoded[j])*xa[b*256+j];
        }
        require(std::isfinite(result[64+r]));double d=result[64+r]-ref;
        chunk_error+=d*d;chunk_energy+=ref*ref;chunk_maximum=std::max(chunk_maximum,std::abs(d));
    }
    if(chunk_energy)require(chunk_error/chunk_energy<1e-10);else require(chunk_error==0);
    for(int j=0;j<64;++j)require(result[j]==-12345 && result[n+64+j]==-12345);
    std::printf("Q6_CHUNK_SCALE_PASS nmse=%.12g maxabs=%.12g\n",chunk_energy?chunk_error/chunk_energy:0,chunk_maximum);
    std::puts("Q6_STANDALONE_PASS");
}
