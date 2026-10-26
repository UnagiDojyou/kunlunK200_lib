from pathlib import Path
import os,sys,json,itertools
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026';sys.path.insert(0,str(O));import runner;runner.OUT=O
os.environ['LD_LIBRARY_PATH']=':'.join(str(R/x) for x in ['external/xre/lib64','external/xdnn/so','external/xtdk/shlib'])
exe=R/'build/flash-decode-lib/k200-flash-decode-bench'
results=[]
def test(name,pd,d,s,hk,h,c,layout='head',pattern=12,batch=1,extra=None):
 env={'KUNLUN_FLASH_DECODING':'on','KUNLUN_FLASH_DECODING_CHUNKS':str(c),'KUNLUN_FLASH_DECODING_LAYOUT':layout,'KUNLUN_FLASH_KQ':'auto','KUNLUN_FLASH_ONLINE':'auto',**(extra or {})}
 text=runner.run(name,[exe,pd,d,s,1,hk,h,pattern,batch],env,180);lines=[l for l in text.splitlines() if l.startswith(('FLASH_UNIT','DECODE_COMPARE','KUNLUN Flash-Decoding'))];print('\n'.join(lines),flush=True)
 if extra and 'KUNLUN_FLASH_DECODING_COMPARE_LAYOUT' in extra:assert 'bitdiff=0' in text
 results.append(dict(name=name,lines=lines));(O/'grid-results.json').write_text(json.dumps(results,indent=2))
for pd,d,hk,s,c in itertools.product([0,1],[64,128],[1,4,32],[256,512,1024,2048,4096],[1,2,4,8]):
 if hk==32 and d==64:continue
 test(f'grid-p{pd}-d{d}-hk{hk}-s{s}-c{c}',pd,d,s,hk,32,c)
for pd,d,hk,s,c in itertools.product([0,1],[64,128],[4,32],[8192,16384],[4,8,16]):
 if hk==32 and d==64:continue
 test(f'long-p{pd}-d{d}-hk{hk}-s{s}-c{c}',pd,d,s,hk,32,c)
for pd in [0,1]:
 for s,c in [(256,2),(2048,2),(2048,4),(4096,8),(16384,16)]:
  test(f'layout-p{pd}-s{s}-c{c}',pd,128,s,4,32,c,extra={'KUNLUN_FLASH_DECODING_COMPARE_LAYOUT':'1'})
 for pattern in [1,3,4,5,6,7,8,9,10,11,13,14,16,17,18,20,21,22,23]:
  s=4096 if pattern==23 else 1056;c=4 if pattern==23 else 8;batch=4 if pattern in [13,18] else 1
  test(f'edge-p{pd}-pat{pattern}',pd,128,s,2,8,c,pattern=pattern,batch=batch,extra={'KUNLUN_FLASH_BENCH_MUTATE':'1'} if pattern in [3,11,18] else None)
 for s,c in [(32,64),(544,2),(4128,4)]:test(f'ragged-p{pd}-s{s}-c{c}',pd,64,s,1,8,c)
 for c in [2,4,8]:
  test(f'trace-p{pd}-c{c}',pd,128,4096,4,32,c,extra={'KUNLUN_FLASH_TRACE':'1'})
print('GRID COMPLETE',len(results),flush=True)
