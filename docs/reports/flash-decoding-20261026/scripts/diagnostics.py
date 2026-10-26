from pathlib import Path
import os,sys
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026';sys.path.insert(0,str(O));import runner;runner.OUT=O
os.environ['LD_LIBRARY_PATH']=':'.join(str(R/x) for x in ['external/xre/lib64','external/xdnn/so','external/xtdk/shlib'])
exe=R/'build/flash-decode-lib/k200-flash-decode-bench'
def run(name,pd,d,s,hk,h,c,pattern=12,b=1,env=None):
 e={'KUNLUN_FLASH_DECODING':'on','KUNLUN_FLASH_DECODING_CHUNKS':str(c),'KUNLUN_FLASH_DECODING_LAYOUT':'head','KUNLUN_FLASH_KQ':'auto','KUNLUN_FLASH_ONLINE':'auto',**(env or {})}
 t=runner.run(name,[exe,pd,d,s,1,hk,h,pattern,b],e,180);print('\n'.join(l for l in t.splitlines() if l.startswith(('FLASH_UNIT','DECODE_COMPARE','KUNLUN Flash'))),flush=True);return t
for pd in [0,1]:
 for hk in [1,4,32]:
  for c in [1,4]:
   run(f'profile-p{pd}-hk{hk}-c{c}',pd,128,4096,hk,32,c,env={'KUNLUN_FLASH_DECODING_PROFILE':'1'})
 for pat in [24,25,27,28]:run(f'mask-p{pd}-pat{pat}',pd,128,4096,4,32,4,pat)
 t=run(f'memory-limit-p{pd}',pd,128,512,1,128,64,b=16);assert 'Flash-Decoding' not in t and 'bitdiff=0' in t
 t=run(f'memory-reserve-p{pd}',pd,128,4096,4,64,8,b=4,env={'KUNLUN_FLASH_DECODE_BENCH_RESERVE_L3_MB':'14'});assert 'Flash-Decoding' not in t and 'bitdiff=0' in t
 for c in [63,64]:run(f'max-chunks-p{pd}-c{c}',pd,128,4096,4,32,c)
 run(f'clamp-odd-p{pd}',pd,128,96,4,32,64)
print('DIAGNOSTICS COMPLETE',flush=True)
