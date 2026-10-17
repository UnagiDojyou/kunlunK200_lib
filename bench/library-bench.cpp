#include <kunlun/k200.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static uint16_t half(float value) {
    uint32_t bits; std::memcpy(&bits,&value,4);
    int exp=int((bits>>23)&255)-127+15;
    kunlun::require(exp > 0 && exp < 31 || value == 0);
    return value == 0 ? 0 : uint16_t(((bits>>16)&0x8000)|(exp<<10)|((bits&0x7fffff)>>13));
}
static void put(int id, void * dst, const void * src, size_t bytes) {
    kunlun::select(id); kunlun::check(kunlun::copy(dst,src,bytes,kunlun::host_to_device)); kunlun::check(kunlun::wait());
}
static void get(int id, void * dst, const void * src, size_t bytes) {
    kunlun::select(id); kunlun::check(kunlun::copy(dst,src,bytes,kunlun::device_to_host)); kunlun::check(kunlun::wait());
}
static int memory(int id) {
    size_t bytes=4<<20;
    std::vector<unsigned char> src(bytes),result(bytes);
    for(size_t i=0;i<bytes;++i) src[i]=(i*131+7)&255;
    kunlun::Buffer a(id,bytes), b(id,bytes), remote(1-id,bytes);
    put(id,a.data(),src.data(),bytes);
    kunlun::select(id); kunlun::check(kunlun::copy(b.data(),a.data(),bytes,kunlun::device_to_device));
    kunlun::peer(1-id,remote.data(),id,b.data(),bytes);
    get(1-id,result.data(),remote.data(),bytes);
    if(src != result) return 1;
    kunlun::peer(id,a.data(),1-id,remote.data(),bytes);
    get(id,result.data(),a.data(),bytes);
    bool ok=src == result;
    std::printf("memory pd=%d bytes=%zu H2D+D2D+P2P+D2H %s\n",id,bytes,ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
int main(int argc, char ** argv) {
    std::lock_guard<std::recursive_mutex> lock(kunlun::runtime_mutex());
    std::string op=argc > 1 ? argv[1] : "matmul", format=argc > 2 ? argv[2] : "f32";
    int id=argc > 3 ? std::atoi(argv[3]) : 0;
    int k=argc > 4 ? std::atoi(argv[4]) : 256, n=argc > 5 ? std::atoi(argv[5]) : 65;
    int batch=argc > 6 ? std::atoi(argv[6]) : 4;
    kunlun::require(id >= 0 && id < 2 && k > 0 && k <= 16384 && k%32 == 0 && n > 0 && n <= 32768 && batch > 0 && batch <= 32);
    if (op == "memory") return memory(id);
    bool pipeline=op == "pipeline";
    if (op != "matmul" && !pipeline) {
        int count=k*batch;
        std::vector<float> a(count),b(k),y(count),gold(count);
        for(int j=0;j<count;++j) a[j]=std::sin(j*.021f);
        for(int j=0;j<k;++j) b[j]=std::cos(j*.04f);
        kunlun::ElementwiseOp kind;
        if(op == "add") kind=kunlun::ElementwiseOp::add;
        else if(op == "multiply") kind=kunlun::ElementwiseOp::multiply;
        else if(op == "silu") kind=kunlun::ElementwiseOp::silu;
        else if(op == "rms") kind=kunlun::ElementwiseOp::rms_norm;
        else if(op == "scale") kind=kunlun::ElementwiseOp::scale;
        else { std::fprintf(stderr,"operation: matmul|pipeline|memory|add|multiply|silu|rms|scale\n"); return 2; }
        float parameter=op == "scale" ? .75f : 1e-5f;
        for(int r=0;r<batch;++r) {
            double sum=0; for(int j=0;j<k;++j) sum+=double(a[r*k+j])*a[r*k+j];
            for(int j=0;j<k;++j) {
                float v=a[r*k+j];
                gold[r*k+j]=op == "add" ? v+b[j] : op == "multiply" ? v*b[j] : op == "silu" ? v/(1+std::exp(-v)) : op == "scale" ? v*parameter : v/std::sqrt(sum/k+parameter);
            }
        }
        kunlun::Buffer da(id,count*4),db(id,k*4),dy(id,count*4);
        kunlun::Kernel kernel;
        put(id,da.data(),a.data(),a.size()*4); put(id,db.data(),b.data(),b.size()*4);
        // RMS also checks the alias scratch path used by graph schedulers.
        void * output=op == "rms" ? da.data() : dy.data();
        kunlun::elementwise(kernel,id,output,da.data(),db.data(),count,k,k,kind,parameter);
        get(id,y.data(),output,y.size()*4);
        float err=0;
        for(int j=0;j<count;++j) { if(!std::isfinite(y[j])) return 1; err=std::max(err,std::abs(y[j]-gold[j])); }
        std::printf("elementwise op=%s pd=%d max_abs=%g %s\n",op.c_str(),id,err,err < 1e-4 ? "PASS" : "FAIL");
        return err < 1e-4 ? 0 : 1;
    }
    kunlun::WeightType type;
    if(format == "f32") type=kunlun::WeightType::f32;
    else if(format == "f16") type=kunlun::WeightType::f16;
    else if(format == "q4_0") type=kunlun::WeightType::q4_0;
    else if(format == "i8") type=kunlun::WeightType::rowwise_i8;
    else { std::fprintf(stderr,"format: f32|f16|q4_0|i8\n"); return 2; }
    std::vector<float> w((size_t)k*n),x((size_t)k*batch),y((size_t)n*batch);
    std::vector<uint16_t> wh(w.size());
    std::vector<unsigned char> q(w.size()/32*18);
    for(int r=0;r<n;++r) for(int j=0;j<k;++j) {
        int v=(r+j)%16;
        w[(size_t)r*k+j]=(v-8)*.0625f;
        wh[(size_t)r*k+j]=half(w[(size_t)r*k+j]);
    }
    for(size_t block=0;block<w.size()/32;++block) {
        uint16_t scale=0x2c00; std::memcpy(q.data()+block*18,&scale,2);
        for(int j=0;j<16;++j) {
            int row=int(block*32/k), col=int(block*32%k)+j;
            q[block*18+2+j]=((row+col)%16)|(((row+col+16)%16)<<4);
        }
    }
    for(size_t j=0;j<x.size();++j) x[j]=std::sin(j*.017f);
    size_t wb=type == kunlun::WeightType::f32 ? w.size()*4 : type == kunlun::WeightType::f16 ? wh.size()*2 : type == kunlun::WeightType::q4_0 ? q.size() : kunlun::packed_i8_size(k,n);
    kunlun::Buffer dw(id,wb),dx(id,x.size()*4),dy(id,y.size()*4);
    kunlun::Kernel kernel;
    if(type == kunlun::WeightType::rowwise_i8) kunlun::upload_i8_rows(id,dw.data(),k,n,0,n,w.data());
    else put(id,dw.data(),type == kunlun::WeightType::f32 ? (void *)w.data() : type == kunlun::WeightType::f16 ? (void *)wh.data() : (void *)q.data(),wb);
    put(id,dx.data(),x.data(),x.size()*4);
    auto run=[&] {
        kunlun::matmul(kernel,id,dy.data(),dw.data(),dx.data(),k,n,batch,type,
            type == kunlun::WeightType::rowwise_i8 ? kunlun::packed_i8_scales(dw.data(),k,n) : nullptr,
            type == kunlun::WeightType::rowwise_i8 ? kunlun::packed_i8_maxima(dw.data(),k,n) : nullptr);
        kunlun::select(id); kunlun::check(kunlun::wait());
        if (pipeline) kunlun::elementwise(kernel,id,dy.data(),dy.data(),dy.data(),n*batch,n,n*batch,kunlun::ElementwiseOp::silu);
    };
    run(); run();
    auto start=std::chrono::steady_clock::now();
    for(int i=0;i<5;++i) run();
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/5;
    get(id,y.data(),dy.data(),y.size()*4);
    double error=0,energy=0,max_abs=0; bool ok=true;
    for(int b=0;b<batch;++b) for(int r=0;r<n;++r) {
        double gold=0; for(int j=0;j<k;++j) gold+=double(w[(size_t)r*k+j])*x[b*k+j];
        if (pipeline) gold=gold/(1+std::exp(-gold));
        double diff=y[b*n+r]-gold;
        double tol=type == kunlun::WeightType::rowwise_i8 ? .02 : .001;
        ok=ok && std::isfinite(y[b*n+r]) && std::abs(diff) <= tol*(1+std::abs(gold));
        max_abs=std::max(max_abs,std::abs(diff)); error+=diff*diff; energy+=gold*gold;
    }
    std::printf("%s format=%s pd=%d k=%d n=%d batch=%d ms=%.6f max_abs=%g nmse=%g %s\n",op.c_str(),format.c_str(),id,k,n,batch,ms,max_abs,error/std::max(energy,1e-30),ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
