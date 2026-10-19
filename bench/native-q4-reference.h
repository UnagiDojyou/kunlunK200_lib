#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <utility>
#include <fstream>

namespace native_q4_bench {
inline void require(bool ok){if(!ok){std::fprintf(stderr,"invalid experiment geometry or nonfinite data\n");std::abort();}}
inline uint32_t random(uint32_t &state){state^=state<<13;state^=state>>17;return state^=state<<5;}
inline float half(const unsigned char *p){_Float16 value;std::memcpy(&value,p,2);return (float)value;}
inline int nibble(const unsigned char *block,int k){return ((block[2+k%16]>>(k<16?0:4))&15)-8;}
inline int signed4(unsigned char value){int v=value&15;return v>=8?v-16:v;}
inline int rounded(float value,int maximum){return std::max(-maximum,std::min(maximum,(int)(value+(value>=0?.5f:-.5f))));}

struct Weights {
    int k,n;
    std::vector<unsigned char> q4,native;
    std::vector<unsigned short> scales;
    std::vector<float> decoded,row_scales,row_inverse;
    Weights(int width,int rows):k(width),n(rows),q4((size_t)n*k/32*18),native(((size_t)n+31)/32*32*k/2),
        scales((size_t)n*k/32),decoded((size_t)n*k),row_scales(n),row_inverse(n){
        require(k>0 && k<=16384 && k%32==0 && n>0 && n>0);
        const char *fixture=std::getenv("KUNLUN_Q4_WEIGHT_FILE");
        if(fixture){std::ifstream f(fixture,std::ios::binary|std::ios::ate);require(f.good() && (size_t)f.tellg()==q4.size());f.seekg(0);f.read((char *)q4.data(),q4.size());require(f.good());}
        uint32_t state=0x7340d123;
        for(int row=0;row<n;++row){
            float maximum=0;
            for(int b=0;b<k/32;++b){
                unsigned char *p=q4.data()+((size_t)row*(k/32)+b)*18;
                if(!fixture){
                    uint32_t h=random(state);float d=std::ldexp((8+(h%9))/1024.f,int((h>>4)%7)-3);
                    if(h&0x80000000u)d=-d;
                    if((row*13+b)%257==0)d=0;
                    _Float16 hd=(_Float16)d;std::memcpy(p,&hd,2);
                    for(int j=0;j<16;++j)p[j+2]=(unsigned char)random(state);
                }
                std::memcpy(&scales[(size_t)row*(k/32)+b],p,2);
                float d=half(p);require(std::isfinite(d));
                for(int j=0;j<32;++j){float w=nibble(p,j)*d;decoded[(size_t)row*k+b*32+j]=w;maximum=std::max(maximum,std::abs(w));}
                for(int j=0;j<16;++j){
                    int lo=nibble(p,2*j),hi=nibble(p,2*j+1);
                    size_t offset=((size_t)b*((n+31)/32)+row/32)*512+j*32+row%32;
                    native[offset]=(unsigned char)((lo&15)|((hi&15)<<4));
                    require(signed4(native[offset])==lo && signed4(native[offset]>>4)==hi);
                }
            }
            row_scales[row]=maximum>0?maximum/127:1;
            row_inverse[row]=maximum>0?127/maximum:0;
        }
    }
};

inline std::vector<float> input(int k,int batch,int pattern=0){
    uint32_t state=0x09200abc;std::vector<float> x((size_t)k*batch);
    for(size_t j=0;j<x.size();++j){
        float value=((int)(random(state)%20001)-10000)/8192.f;
        if(pattern==3)value=(float)((double)(int32_t)random(state)/2147483648.0);
        if(pattern==4){uint32_t bits=random(state);bits=(bits&0x807fffffu)|((107+(random(state)%32))<<23);std::memcpy(&value,&bits,4);}
        if(pattern==1 && j%113==0)value*=1000;
        if(pattern==2)value=0;
        x[j]=value;
    }
    return x;
}
struct Activation {
    int bits,components,k,batch;
    std::vector<int8_t> linear,padded;
    std::vector<float> represented;
    float scale[3]={1,1,1};
    Activation(const std::vector<float> &x,int width,int columns,int precision,int parts,bool need_padded=true,bool reconstruct=true):
        bits(precision),components(parts),k(width),batch(columns),linear(x.size()*parts),
        padded(need_padded?(size_t)parts*(k/32)*32*(bits==4?16:32):0),represented(reconstruct?x.size():0,0){
        require((bits==4 || bits==8) && components>=1 && components<=3 && batch>=1 && batch<=32);
        std::vector<float> residual=x;int limit=bits==4?7:127;
        for(int part=0;part<components;++part){
            float maximum=0;for(float v:residual){require(std::isfinite(v));maximum=std::max(maximum,std::abs(v));}
            if(maximum==0)maximum=1;
            scale[part]=maximum/limit;float inverse=limit/maximum;
            for(size_t j=0;j<x.size();++j){
                int q=rounded(residual[j]*inverse,limit);linear[part*x.size()+j]=(int8_t)q;
                residual[j]-=q*scale[part];if(reconstruct)represented[j]+=q*scale[part];
            }
            if(need_padded)for(int block=0;block<k/32;++block)for(int b=0;b<batch;++b){
                size_t base=((size_t)part*(k/32)+block)*32*(bits==4?16:32)+b*(bits==4?16:32);
                for(int j=0;j<(bits==4?16:32);++j){
                    if(bits==4){int a=linear[part*x.size()+(size_t)b*k+block*32+2*j];int c=linear[part*x.size()+(size_t)b*k+block*32+2*j+1];padded[base+j]=(int8_t)((a&15)|((c&15)<<4));}
                    else padded[base+j]=linear[part*x.size()+(size_t)b*k+block*32+j];
                }
            }
        }
    }
};
inline std::vector<double> reference(const std::vector<float>&w,const std::vector<float>&x,int k,int n,int batch){
    std::vector<double> result((size_t)n*batch);
    #pragma omp parallel for num_threads(6)
    for(int row=0;row<n;++row)for(int b=0;b<batch;++b){
        double sum=0;
        #pragma omp simd reduction(+:sum)
        for(int j=0;j<k;++j)sum+=(double)w[(size_t)row*k+j]*(double)x[(size_t)b*k+j];
        result[(size_t)b*n+row]=sum;
    }
    return result;
}
struct Error {double max_abs=0,max_rel=0,nmse=0;};
template<class T> inline Error error(const std::vector<double>&reference,const std::vector<T>&actual){
    require(reference.size()==actual.size());Error result;double squared=0,energy=0;
    for(size_t i=0;i<reference.size();++i){require(std::isfinite(actual[i]));double e=std::abs(reference[i]-actual[i]);
        result.max_abs=std::max(result.max_abs,e);result.max_rel=std::max(result.max_rel,e/std::max(std::abs(reference[i]),1e-20));squared+=e*e;energy+=reference[i]*reference[i];}
    result.nmse=squared/std::max(energy,1e-30);return result;
}
}
