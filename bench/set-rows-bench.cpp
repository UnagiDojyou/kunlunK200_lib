#include <kunlun/k200.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

static unsigned short cpu_half(float value) {
    _Float16 h=(_Float16)value; unsigned short bits; std::memcpy(&bits,&h,2); return bits;
}
int main(int argc,char **argv) {
    int pd=argc>1?std::atoi(argv[1]):0,width=argc>2?std::atoi(argv[2]):1;
    int rows=argc>3?std::atoi(argv[3]):4,capacity=argc>4?std::atoi(argv[4]):8;
    int source_bytes=argc>5?std::atoi(argv[5]):4,index_bytes=argc>6?std::atoi(argv[6]):8;
    int gapped=argc>7?std::atoi(argv[7]):0,invalid=argc>8?std::atoi(argv[8]):0;
    kunlun::require(pd>=0 && pd<2 && width>0 && width<=16384 && rows>0 && rows<=capacity && capacity<=4194304);
    kunlun::require(source_bytes==2 || source_bytes==4);kunlun::require(index_bytes==4 || index_bytes==8);
    kunlun::SetRowsLayout p{};
    p.width=width;p.rows=rows;p.dim2=gapped?2:1;p.dim3=gapped?2:1;
    p.source_bytes=source_bytes;p.index_bytes=index_bytes;p.destination_rows=capacity;
    p.source_stride[0]=source_bytes*(gapped?2:1);
    p.source_stride[1]=width*p.source_stride[0]+(gapped?32:0);
    p.source_stride[2]=rows*p.source_stride[1]+(gapped?32:0);
    p.source_stride[3]=p.dim2*p.source_stride[2]+(gapped?32:0);
    p.destination_stride[0]=width*2+(gapped?32:0);
    p.destination_stride[1]=capacity*p.destination_stride[0]+(gapped?32:0);
    p.destination_stride[2]=p.dim2*p.destination_stride[1]+(gapped?32:0);
    p.index_stride[0]=index_bytes*(gapped?2:1);p.index_stride[1]=rows*p.index_stride[0];p.index_stride[2]=p.index_stride[1];
    p.index_extent[0]=p.index_extent[1]=1;
    int features=argc>9?std::atoi(argv[9]):0;
    if(features) {
        kunlun::require(width==1 && !gapped && rows%features==0 && capacity%features==0);
        p.cache_features=features;p.cache_tokens=rows/features;p.cache_stride=capacity/features;
    }
    size_t src_bytes=(size_t)p.dim3*p.source_stride[3],dst_bytes=(size_t)p.dim3*p.destination_stride[2],idx_bytes=rows*p.index_stride[0];
    size_t guard=gapped?34:32;
    std::vector<unsigned char>src(src_bytes,0x5a),idx(idx_bytes,0x5a),initial(dst_bytes+guard+64,0xa5),gold=initial,actual(initial.size());
    for(int row=0;row<rows;++row) {
        int64_t target=features?(int64_t)(row%features)*p.cache_stride+3+row/features:capacity-1-row;
        if(invalid && row==0)target=-1;
        if(invalid && row==1)target=index_bytes==8?(int64_t(1)<<32):capacity+1;
        std::memcpy(idx.data()+row*p.index_stride[0],&target,index_bytes);
        for(int i3=0;i3<p.dim3;++i3)for(int i2=0;i2<p.dim2;++i2)for(int j=0;j<width;++j) {
            float value=(int((j+row*13+i2*7+i3*11)%257)-128)*.03125f;
            if(j==0 && row==0)value=-0.0f;
            if(j==0 && row==1)value=std::ldexp(1.0f,-24);
            if(j==0 && row==2)value=std::ldexp(1.0f,-25);
            if(j==0 && row==3)value=std::numeric_limits<float>::infinity();
            if(j==0 && row==4)value=std::numeric_limits<float>::quiet_NaN();
            size_t so=(size_t)i3*p.source_stride[3]+i2*p.source_stride[2]+row*p.source_stride[1]+j*p.source_stride[0];
            unsigned short h=cpu_half(value);
            std::memcpy(src.data()+so,source_bytes==4?(const void *)&value:(const void *)&h,source_bytes);
            if(target>=0 && target<capacity) {
                size_t dest=guard+(size_t)i3*p.destination_stride[2]+i2*p.destination_stride[1]+target*p.destination_stride[0]+j*2;
                std::memcpy(gold.data()+dest,&h,2);
            }
        }
    }
    kunlun::Buffer ds(pd,src.size()),di(pd,idx.size()),dy(pd,initial.size());kunlun::Kernel kernel;
    kunlun::select(pd);
    kunlun::check(kunlun::copy(ds.data(),src.data(),src.size(),kunlun::host_to_device));
    kunlun::check(kunlun::copy(di.data(),idx.data(),idx.size(),kunlun::host_to_device));
    kunlun::check(kunlun::copy(dy.data(),initial.data(),initial.size(),kunlun::host_to_device));kunlun::check(kunlun::wait());
    auto run=[&]{return kunlun::set_rows_f16(kernel,pd,(char *)dy.data()+guard,dst_bytes,ds.data(),src_bytes,di.data(),idx_bytes,p);};
    bool valid=run();
    bool expect_valid=!invalid;
    if(valid!=expect_valid){std::fprintf(stderr,"index status mismatch\n");return 1;}
    kunlun::check(kunlun::copy(actual.data(),dy.data(),actual.size(),kunlun::device_to_host));kunlun::check(kunlun::wait());
    if(actual!=gold){for(size_t j=0;j<gold.size();++j)if(actual[j]!=gold[j]){std::fprintf(stderr,"FAIL byte=%zu got=%u ref=%u\n",j,actual[j],gold[j]);break;}return 1;}
    for(int repeat=0;repeat<10 && !invalid;++repeat) {
        kunlun::check(kunlun::copy(dy.data(),initial.data(),initial.size(),kunlun::host_to_device));kunlun::check(kunlun::wait());
        if(!run())return 1;
        kunlun::check(kunlun::copy(actual.data(),dy.data(),actual.size(),kunlun::device_to_host));kunlun::check(kunlun::wait());
        if(actual!=gold){std::fprintf(stderr,"repeated update mismatch repeat=%d\n",repeat);return 1;}
    }
    std::vector<double>times;
    if(!invalid)for(int g=0;g<5;++g){auto start=std::chrono::steady_clock::now();for(int i=0;i<20;++i)if(!run())return 1;
        times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/20);}
    std::sort(times.begin(),times.end());
    std::printf("set_rows pd=%d width=%d rows=%d capacity=%d source_bytes=%d index_bytes=%d gapped=%d invalid=%d median_ms=%.6f bitwise_equal=1 PASS\n",pd,width,rows,capacity,source_bytes,index_bytes,gapped,invalid,times.empty()?0:times[2]);
}
