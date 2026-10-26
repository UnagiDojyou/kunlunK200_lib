from pathlib import Path
import os,sys,json,subprocess
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026';sys.path.insert(0,str(O));import runner;runner.OUT=O
os.environ['LD_LIBRARY_PATH']=':'.join(str(R/x) for x in ['external/xre/lib64','external/xdnn/so','external/xtdk/shlib'])
# Run only after normal performance jobs: expose non-power-of-two experimental
# chunk counts without changing the default dispatch or default chunk choice.
p=R/'repos/KunlunK200/src/flash-decode.inc';s=p.read_text();s=s.replace('n==1||n==2||n==4||n==8||n==16||n==32||n==64','n>=1&&n<=64');p.write_text(s)
env=os.environ.copy();env.update(XRE=str(R/'external/xre'),XDNN=str(R/'external/xdnn'),XTDK_DIR=str(R/'external/xtdk'))
subprocess.run(['cmake','--build',str(R/'build/flash-decode-lib'),'-j2','--target','k200-flash-decode-bench'],env=env,check=True)
exe=R/'build/flash-decode-lib/k200-flash-decode-bench';rows=[]
def test(name,pd,d,s,hk,h=32,c=None,m=1,e=None):
 env={'KUNLUN_FLASH_DECODING':'on','KUNLUN_FLASH_KQ':'auto','KUNLUN_FLASH_ONLINE':'auto',**(e or {})}
 if c:env['KUNLUN_FLASH_DECODING_CHUNKS']=str(c)
 t=runner.run(name,[exe,pd,d,s,m,hk,h,14,1],env,180);line=next(l for l in t.splitlines() if l.startswith('FLASH_UNIT'));print(line,flush=True);rows.append(dict(test=name,line=line));(O/'extra-units.json').write_text(json.dumps(rows,indent=2))
 if m>1:assert 'Flash-Decoding' not in t and 'bitdiff=0' in t
for pd in [0,1]:
 for d,hk in [(64,4),(128,32)]:
  for s in [1280,2304,4352]:
   for c in [2,3,4,5,8]:test(f'padded-p{pd}-d{d}-hk{hk}-s{s}-c{c}',pd,d,s,hk,c=c)
 for s in [256,512,1024,2048,4096]:test(f'extra-mha-p{pd}-s{s}',pd,64,s,32,c=4)
 for length in [256,512,1024]:test(f'extra-length-p{pd}-l{length}',pd,128,4096,4,e={'KUNLUN_FLASH_DECODING_CHUNK':str(length)})
 for block in [256,512]:test(f'extra-block-p{pd}-b{block}',pd,128,2048,4,c=2,e={'KUNLUN_FLASH_BLOCK':str(block)})
 for hk in [1,4]:test(f'extra-ungroup-p{pd}-hk{hk}',pd,128,4096,hk,c=4,e={'KUNLUN_FLASH_TG_GROUP':'0'})
 test(f'extra-hbm-p{pd}',pd,128,2048,4,c=4,e={'KUNLUN_FLASH_HBM':'1'})
 test(f'extra-pp-p{pd}',pd,64,512,4,m=128)
print('EXTRA COMPLETE',flush=True)
