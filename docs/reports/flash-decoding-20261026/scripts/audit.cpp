#include <xpu/runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <mutex>
#include <chrono>
struct Value{unsigned long long count=0,bytes=0;double ms=0;};
struct State{std::recursive_mutex mutex;std::map<std::string,Value>values;std::map<void*,std::pair<int,uint64_t>>mem;uint64_t live[2][2]={},peak[2][2]={};int pd=0;};
static State &s(){static State *p=new State;return *p;}
template<class F>F next(const char *n){void *p=dlsym(RTLD_NEXT,n);if(!p)std::abort();return reinterpret_cast<F>(p);}
static void add(const char *name,uint64_t bytes,double ms=0){const char *phase=getenv("K200_PROFILE_PHASE");std::lock_guard<std::recursive_mutex>lock(s().mutex);auto &v=s().values[std::string(phase?phase:"other")+"|"+name+"|pd"+std::to_string(s().pd)];++v.count;v.bytes+=bytes;v.ms+=ms;}
extern "C" int xpu_set_device(int pd){static auto fn=next<int(*)(int)>("xpu_set_device");int rc=fn(pd);if(!rc)s().pd=pd;return rc;}
extern "C" int xpu_memcpy(void *d,const void *p,uint64_t n,XPUMemcpyKind k){static auto fn=next<int(*)(void*,const void*,uint64_t,XPUMemcpyKind)>("xpu_memcpy");add(k==XPU_HOST_TO_DEVICE?"H2D":k==XPU_DEVICE_TO_HOST?"D2H":"D2D",n);return fn(d,p,n,k);}
extern "C" int xpu_launch_async(XPUFunc f){static auto fn=next<int(*)(XPUFunc)>("xpu_launch_async");add("launch",0);return fn(f);}
extern "C" int xpu_wait(XPUStream st){static auto fn=next<int(*)(XPUStream)>("xpu_wait");auto t=std::chrono::steady_clock::now();int rc=fn(st);add("wait",0,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());return rc;}
extern "C" int xpu_malloc(void **p,uint64_t n,XPUMemoryKind kind){static auto fn=next<int(*)(void**,uint64_t,XPUMemoryKind)>("xpu_malloc");int rc=fn(p,n,kind);if(!rc){std::lock_guard<std::recursive_mutex>lock(s().mutex);int k=kind==XPU_MEM_L3?1:0;s().mem[*p]={s().pd*2+k,n};s().live[s().pd][k]+=n;s().peak[s().pd][k]=std::max(s().peak[s().pd][k],s().live[s().pd][k]);}return rc;}
extern "C" int xpu_free(void *p){static auto fn=next<int(*)(void*)>("xpu_free");int rc=fn(p);if(!rc){std::lock_guard<std::recursive_mutex>lock(s().mutex);auto it=s().mem.find(p);if(it!=s().mem.end()){int tag=it->second.first;s().live[tag/2][tag%2]-=it->second.second;s().mem.erase(it);}}return rc;}
extern "C" void k200_profile_emit(){std::lock_guard<std::recursive_mutex>lock(s().mutex);for(auto &v:s().values)fprintf(stderr,"FLASH_AUDIT key=%s count=%llu bytes=%llu wait_ms=%.6f\n",v.first.c_str(),v.second.count,v.second.bytes,v.second.ms);s().values.clear();}
extern "C" void k200_profile_memory(){for(int pd=0;pd<2;++pd)fprintf(stderr,"FLASH_MEMORY pd=%d hbm_peak=%llu l3_peak=%llu hbm_live=%llu l3_live=%llu\n",pd,(unsigned long long)s().peak[pd][0],(unsigned long long)s().peak[pd][1],(unsigned long long)s().live[pd][0],(unsigned long long)s().live[pd][1]);}
__attribute__((destructor))static void report(){k200_profile_emit();k200_profile_memory();}
