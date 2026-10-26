from pathlib import Path
import os,subprocess
R=Path('/home/unagidojyou/kunlun-k200');L=R/'repos/KunlunK200'
p=L/'bench/flash-decode-bench.cpp';s=p.read_text()
s=s.replace('setenv("KUNLUN_FLASH_DECODING","off",1);require(', 'bool compare_layout=getenv("KUNLUN_FLASH_DECODING_COMPARE_LAYOUT");setenv("KUNLUN_FLASH_DECODING",compare_layout?"on":"off",1);if(compare_layout)setenv("KUNLUN_FLASH_DECODING_LAYOUT","chunk",1);require(')
s=s.replace('setenv("KUNLUN_FLASH_DECODING",mode.c_str(),1);};','setenv("KUNLUN_FLASH_DECODING",mode.c_str(),1);if(compare_layout)setenv("KUNLUN_FLASH_DECODING_LAYOUT","head",1);};')
# Repeated scores keep the block maximum fixed, isolating chunk-sum rounding
# from the change in probability quantization caused by local maxima.
s=s.replace('cosf(x*.19f+r*.11f+j*.2f)','cosf(x*.19f+(pattern==23?r%1024:r)*.11f+j*.2f)')
p.write_text(s)
p=L/'src/flash.inc';s=p.read_text().replace('size_t workspace=0,calls=0;','size_t workspace=0,calls=0,decode_calls=0;');p.write_text(s)
p=L/'src/flash-decode.inc';s=p.read_text().replace('if(!s.calls++||trace)std::fprintf','++s.calls;if(!s.decode_calls++||trace)std::fprintf').replace('storage=%s layout=%s calls=%zu','storage=%s pool=%zu layout=%s calls=%zu').replace('hbm?"HBM":"L3",head?', 'hbm?"HBM":"L3",hbm?0:s.l3->bytes,head?');p.write_text(s)
env=os.environ.copy();env.update(XRE=str(R/'external/xre'),XDNN=str(R/'external/xdnn'),XTDK_DIR=str(R/'external/xtdk'))
subprocess.run(['cmake','--build',str(R/'build/flash-decode-lib'),'-j6','--target','k200-flash-decode-bench'],env=env,check=True)
