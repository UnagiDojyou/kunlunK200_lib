#include <kunlun/k200.h>
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace kunlun;
static double ms(std::chrono::steady_clock::time_point t){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();}
int main(int argc,char **argv) {
    int k=argc>1?atoi(argv[1]):4096,pd=argc>2?atoi(argv[2]):0;
    auto *backend=ggml_backend_cpu_init();require(backend);ggml_backend_cpu_set_n_threads(backend,6);
    auto params=ggml_threadpool_params_default(6);auto pool=ggml_threadpool_new(&params);require(pool);ggml_backend_cpu_set_threadpool(backend,pool);
    Kernel kernel;
    for(int shape=0;shape<4;++shape) {
        int n=shape==0?1:shape==3?512:128,m=shape<2?1:n;
        std::vector<uint32_t>input((size_t)k*n),host(input.size()),reference((size_t)k*m),out(reference.size());std::vector<int>indices(m);
        for(size_t j=0;j<input.size();++j)input[j]=(uint32_t)(j*1664525u+1013904223u);
        for(int j=0;j<m;++j)indices[j]=j%3==0?n-1:j%3==1?0:(j*17)%n;
        Buffer x(pd,input.size()*4),y(pd,out.size()*4),ids(pd,indices.size()*4);
        check(copy(x.data(),input.data(),input.size()*4,host_to_device));check(copy(ids.data(),indices.data(),m*4,host_to_device));host=input;
        auto*c=ggml_init({1024*1024,nullptr,true});auto *rx=ggml_new_tensor_2d(c,GGML_TYPE_F32,k,n);rx->data=host.data();auto *ri=ggml_new_tensor_1d(c,GGML_TYPE_I32,m);ri->data=indices.data();auto *ry=ggml_get_rows(c,rx,ri);ry->data=reference.data();auto *graph=ggml_new_graph(c);ggml_build_forward_expand(graph,ry);
        auto cpu=[&]{require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);};
        auto gpu=[&]{require(get_rows_f32(kernel,pd,y.data(),out.size()*4,x.data(),input.size()*4,(const int32_t*)ids.data(),m*4,k,n,m));};
        auto d2h=[&]{check(copy(host.data(),x.data(),input.size()*4,device_to_host));check(wait());};
        auto h2d=[&]{check(copy(y.data(),reference.data(),reference.size()*4,host_to_device));check(wait());};
        auto idx=[&]{check(copy(ids.data(),indices.data(),m*4,host_to_device));check(wait());};
        cpu();gpu();check(copy(out.data(),y.data(),out.size()*4,device_to_host));require(out==reference);
        std::vector<double>a,b,dt,ht,it,oldcost,newcost;
        for(int r=0;r<3;++r) {
            auto t=std::chrono::steady_clock::now();cpu();a.push_back(ms(t));t=std::chrono::steady_clock::now();gpu();b.push_back(ms(t));
            t=std::chrono::steady_clock::now();d2h();dt.push_back(ms(t));t=std::chrono::steady_clock::now();h2d();ht.push_back(ms(t));t=std::chrono::steady_clock::now();idx();it.push_back(ms(t));
            t=std::chrono::steady_clock::now();d2h();cpu();h2d();oldcost.push_back(ms(t));t=std::chrono::steady_clock::now();idx();gpu();newcost.push_back(ms(t));
        }
        auto med=[](std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];};
        printf("F32_CPU_XPU K=%d N=%d M=%d bitwise=1 cpu_ms=%.6f gpu_ms=%.6f d2h_ms=%.6f h2d_ms=%.6f idx_ms=%.6f old_total_ms=%.6f new_total_ms=%.6f\n",k,n,m,med(a),med(b),med(dt),med(ht),med(it),med(oldcost),med(newcost));fflush(stdout);ggml_free(c);
    }
    ggml_backend_cpu_set_threadpool(backend,nullptr);ggml_threadpool_free(pool);ggml_backend_free(backend);puts("F32_GGML_REFERENCE_PASS");
}
