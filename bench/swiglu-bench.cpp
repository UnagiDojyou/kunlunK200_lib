#include <kunlun/k200.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

static float load(const unsigned char *p,int bytes) {
    if(bytes==4){float x;std::memcpy(&x,p,4);return x;}
    _Float16 h;std::memcpy(&h,p,2);return (float)h;
}
static void store(unsigned char *p,int bytes,float x) {
    if(bytes==4)std::memcpy(p,&x,4);
    else{_Float16 h=(_Float16)x;std::memcpy(p,&h,2);}
}
int main(int argc,char **argv) {
    int pd=argc>1?std::atoi(argv[1]):0,w=argc>2?std::atoi(argv[2]):11008;
    int rows=argc>3?std::atoi(argv[3]):128,bytes=argc>4?std::atoi(argv[4]):4;
    int pattern=argc>5?std::atoi(argv[5]):0,packed=argc>6?std::atoi(argv[6]):0;
    int gap=argc>7?std::atoi(argv[7]):0,alias=argc>8?std::atoi(argv[8]):0;
    kunlun::require(pd>=0 && pd<2 && w>0 && w<=65536 && rows>0 && rows<=1024);
    kunlun::require((bytes==2 || bytes==4) && pattern>=0 && pattern<=5 && alias>=0 && alias<=2);
    int gs=w*bytes*(packed?2:1)+gap,us=gs,ys=w*bytes+gap;
    if(alias)ys=gs;
    kunlun::SwiGLULayout p{w,rows,gs,us,ys,bytes==4?kunlun::InputType::f32:kunlun::InputType::f16};
    kunlun::require(!*kunlun::swiglu_reason(p));
    size_t guard=128,span=(size_t)(rows-1)*gs+w*bytes;
    size_t gsize=span+(packed?w*bytes:0)+2*guard,ysize=(size_t)(rows-1)*ys+w*bytes+2*guard;
    std::vector<unsigned char>g(gsize,0xa5),u(gsize,0x5a),y(ysize,0xa5);
    for(int r=0;r<rows;++r)for(int j=0;j<w;++j) {
        unsigned hash=(unsigned)(j+r*w)*1664525u+1013904223u;
        float gv=((int)((hash>>8)%4097)-2048)/256.f,uv=((int)((hash>>19)%1025)-512)/128.f;
        if(pattern==1)gv=uv=0;
        if(pattern==2 && j%61==0)gv=j%122?-200.f:200.f;
        if(pattern==3)gv=(j%15-7)*.125f;
        if(pattern==4 && r==0 && j<8) {
            const float special[]={-0.f,0.f,std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN(),1e-8f,-1e-8f,1.f};gv=special[j];
        }
        if(j==0 && !pattern)gv=-1; if(j==1 && !pattern)gv=0;if(j==2 && !pattern)gv=1;
        if(pattern==5 && r==0 && j<8) {
            const float inf=std::numeric_limits<float>::infinity(),nan=std::numeric_limits<float>::quiet_NaN();
            const float gx[]={0.f,-0.f,1.f,-1.f,-200.f,inf,-inf,1.f};
            const float ux[]={inf,-inf,inf,inf,inf,0.f,0.f,nan};gv=gx[j];uv=ux[j];
        }
        store(g.data()+guard+(size_t)r*gs+j*bytes,bytes,gv);
        store((packed?g.data()+w*bytes:u.data())+guard+(size_t)r*us+j*bytes,bytes,uv);
    }
    auto initial=alias==1?g:alias==2?(packed?g:u):y;
    std::vector<unsigned char>actual(initial.size());std::vector<bool>written(initial.size());
    std::vector<float>reference((size_t)w*rows);
    size_t outoff=guard+(alias==2 && packed?w*bytes:0);
    for(int r=0;r<rows;++r)for(int j=0;j<w;++j) {
        float gv=load(g.data()+guard+(size_t)r*gs+j*bytes,bytes);
        float uv=load((packed?g.data()+w*bytes:u.data())+guard+(size_t)r*us+j*bytes,bytes);
        float expected=(gv/(1.f+std::exp(-gv)))*uv;
        if(bytes==2)expected=(float)(_Float16)expected;
        reference[(size_t)r*w+j]=expected;
        for(int b=0;b<bytes;++b)written[outoff+(size_t)r*ys+j*bytes+b]=true;
    }
    kunlun::Buffer dg(pd,g.size()),du(pd,u.size()),dy(pd,y.size());kunlun::Kernel k;
    kunlun::select(pd);
    kunlun::check(kunlun::copy(dg.data(),g.data(),g.size(),kunlun::host_to_device));
    kunlun::check(kunlun::copy(du.data(),u.data(),u.size(),kunlun::host_to_device));
    kunlun::check(kunlun::copy(dy.data(),y.data(),y.size(),kunlun::host_to_device));kunlun::check(kunlun::wait());
    const void *gp=(char*)dg.data()+guard,*up=(packed?(char*)dg.data()+w*bytes:(char*)du.data())+guard;
    void *output=alias==1?(void*)gp:alias==2?(void*)up:(char*)dy.data()+guard;
    auto run=[&]{return kunlun::swiglu(k,pd,output,initial.size()-outoff,gp,g.size()-guard,up,u.size()-guard,p);};
    if(!run()){std::fprintf(stderr,"unexpected API rejection\n");return 2;}
    const void *base=alias==1?dg.data():alias==2?(packed?dg.data():du.data()):dy.data();
    kunlun::check(kunlun::copy(actual.data(),base,actual.size(),kunlun::device_to_host));kunlun::check(kunlun::wait());
    double sq=0,ref_sq=0,max_abs=0,max_ref=0;int special_errors=0,padding_errors=0;
    for(size_t i=0;i<actual.size();++i)if(!written[i] && actual[i]!=initial[i])++padding_errors;
    for(int r=0;r<rows;++r)for(int j=0;j<w;++j) {
        float a=load(actual.data()+outoff+(size_t)r*ys+j*bytes,bytes),b=reference[(size_t)r*w+j];
        if(!std::isfinite(b)){if((std::isnan(b)&&!std::isnan(a)) || (std::isinf(b)&&a!=b))++special_errors;continue;}
        if(!std::isfinite(a)){++special_errors;continue;}
        double d=double(a)-b;sq+=d*d;ref_sq+=double(b)*b;max_abs=std::max(max_abs,std::abs(d));max_ref=std::max(max_ref,std::abs(double(b)));
    }
    double nmse=sq/std::max(ref_sq,1e-30);bool pass=!special_errors && !padding_errors && nmse<(bytes==4?1e-12:1e-6) && max_abs<(bytes==4?2e-5:2e-3)*(1+max_ref);
    std::vector<double>times;
    if(!alias && pass) {
        if(!run())return 2;
        for(int t=0;t<15;++t){auto start=std::chrono::steady_clock::now();if(!run())return 2;times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());}
        std::sort(times.begin(),times.end());
    }
    std::printf("SWIGLU pd=%d width=%d rows=%d bytes=%d pattern=%d packed=%d gap=%d alias=%d nmse=%.12g max_abs=%.12g special_errors=%d padding_errors=%d median_ms=%.6f min_ms=%.6f max_ms=%.6f pass=%d\n",pd,w,rows,bytes,pattern,packed,gap,alias,nmse,max_abs,special_errors,padding_errors,times.empty()?0:times[7],times.empty()?0:times.front(),times.empty()?0:times.back(),pass);
    for(int j=0;j<std::min(w,pattern>=4?8:3);++j)std::printf("SAMPLE j=%d actual=%.9g reference=%.9g\n",j,load(actual.data()+outoff+j*bytes,bytes),reference[j]);
    return 0;
}
