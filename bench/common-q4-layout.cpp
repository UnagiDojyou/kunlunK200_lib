#include "../src/q4-common-pack.h"
#include <cstdio>
#include <fstream>
using namespace common_q4;
int main(){
    size_t cases=0,checked=0;std::ofstream map("byte-map.csv");map<<"layout,scale_order,N,K,pair,bank,word_row,byte_offset,scale_offset,l1w_vector_for_Kt128_Nt64,l1w_bank,byte_lane\n";
    for(auto dims:std::vector<std::pair<int,int>>{{33,33},{512,512},{11008,129},{16383,65}}){int k=dims.first,n=dims.second,b=(k+31)/32;std::vector<unsigned char> raw(size_t(n)*b*18);
        for(size_t i=0;i<raw.size();++i)raw[i]=(i*131+(i>>8))&255;
        for(int l=0;l<3;++l)for(int so=0;so<2;++so){Pack p(raw.data(),raw.size(),k,n,Layout(l),ScaleOrder(so));Pack fast(raw.data(),raw.size(),k,n,Layout(l),ScaleOrder(so),false);if(fast.bytes!=p.bytes)throw std::runtime_error("unchecked pack byte mismatch");std::vector<unsigned char> seen(p.info.scale_offset,0);
            for(int c=0;c<p.info.np;++c)for(int pair=0;pair<p.info.kp/2;++pair){size_t a=weight_index(p.info,c,pair);if(a>=seen.size()||seen[a]++)throw std::runtime_error("weight mapping not bijective");++checked;}
            if(p.bytes.size()*8!=size_t(p.info.np)*p.info.kp*9/2)throw std::runtime_error("4.5 padded bpw");
            for(int c:{0,15,16,31,32})for(int j:{0,1,2,15,16,17,31,32,127,128}){
                if(c>=p.info.np||j>=p.info.kp)continue;int pair=j/2,row=(c%32/16)*8+(pair%16)/2;
                map<<l<<','<<so<<','<<c<<','<<j<<','<<pair<<','<<c%16<<','<<row<<','<<weight_index(p.info,c,pair)<<','<<scale_index(p.info,c,j/32)<<','<<(c%64/32)*64+(j%128/32)*16+row<<','<<c%16<<','<<pair%2<<'\n';
            }
            ++cases;
        }
    }
    // Exercise every 16-bit FP16 bit pattern without numerical conversion.
    std::vector<unsigned char> raw(65536*18,0x88);for(int i=0;i<65536;++i){raw[i*18]=i&255;raw[i*18+1]=i>>8;}
    for(int l=0;l<3;++l)for(int so=0;so<2;++so){Pack p(raw.data(),raw.size(),16384,128,Layout(l),ScaleOrder(so));++cases;}
    std::printf("CPU_LAYOUT {\"cases\":%zu,\"mapped_bytes\":%zu,\"all_fp16_bits\":true,\"reversible\":true,\"padded_bpw\":4.5}\n",cases,checked);
}
