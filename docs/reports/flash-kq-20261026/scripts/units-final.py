from pathlib import Path
import os,sys,json,re,itertools
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-kq-20261026';sys.path.insert(0,str(O));import runner;runner.OUT=O
os.environ['LD_LIBRARY_PATH']=':'.join(str(R/x) for x in ['external/xre/lib64','external/xdnn/so','external/xtdk/shlib'])
rows=[]
def unit(name,pd,d,s,m,hk,h,pattern,b=1,mutate=False):
 env={'KUNLUN_FLASH_KQ':'reuse1024','KUNLUN_FLASH_BENCH_SAMPLE':'1'}
 if mutate:env['KUNLUN_FLASH_BENCH_MUTATE']='1'
 text=runner.run(name,[R/'build/flash-kq-lib/k200-flash-kq-api-bench',pd,d,s,m,hk,h,pattern,b],env,180)
 lines=[x for x in text.splitlines() if x.startswith(('FLASH_UNIT','KQ_COMPARE','FLASH_MUTATE'))];print('\n'.join(lines),flush=True)
 assert 'bitdiff=0' in text and 'pass=1' in text
 rows.append(dict(name=name,lines=lines));(O/'units-final.json').write_text(json.dumps(rows,indent=2))
for pd,d,s,m,pattern in itertools.product([0,1],[64,128],[256,512,2048,4096],[1,128],[0,12]):
 unit(f'grid-p{pd}-d{d}-s{s}-m{m}-pat{pattern}',pd,d,s,m,4 if d==64 else 8,32,pattern)
for pd in [0,1]:
 for pattern in [1,3,4,5,6,7,8,9,10,11,13,14,16,17,18,20,21,22]:
  m=1 if pattern==17 else 3;b=4 if pattern in [13,18] else 1
  unit(f'edge-p{pd}-pat{pattern}',pd,128,544,m,2,8,pattern,b,pattern in [3,11,18])
 unit(f'mutate-real-p{pd}',pd,128,2048,128,8,32,11,1,True)
print('FINAL UNITS COMPLETE',len(rows),flush=True)

