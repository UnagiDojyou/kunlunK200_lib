from pathlib import Path
import os,sys,json,time,subprocess,runpy
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026'
deadline=time.monotonic()+5400
while not (O/'AB_DONE').exists():
 if (O/'XPU_ABORT').exists():raise RuntimeError('stop latch')
 if time.monotonic()>deadline:raise RuntimeError('AB timeout')
 time.sleep(5)
env=os.environ.copy();env.update(XRE=str(R/'external/xre'),XDNN=str(R/'external/xdnn'),XTDK_DIR=str(R/'external/xtdk'))
subprocess.run(['cmake','--build',str(R/'build/flash-decode-lib'),'-j4','--target','k200-flash-decode-bench'],env=env,check=True)
for script in ['grid.py','diagnostics.py']:
 print('BEGIN',script,flush=True);runpy.run_path(str(O/script),run_name='__main__');print('END',script,flush=True)
sys.path.insert(0,str(O));import runner;runner.OUT=O
os.environ['LD_LIBRARY_PATH']=':'.join(str(R/x) for x in ['external/xre/lib64','external/xdnn/so','external/xtdk/shlib'])
exe=R/'build/flash-decode-lib/k200-flash-decode-bench'
for pd in [0,1]:
 for d,hk in [(64,4),(128,32),(128,1)]:
  for c in [3,5]:
   t=runner.run(f'final-odd-mask-p{pd}-d{d}-hk{hk}-c{c}',[exe,pd,d,4352,1,hk,32,12,1],{'KUNLUN_FLASH_DECODING':'on','KUNLUN_FLASH_DECODING_CHUNKS':str(c)},180);print(next(l for l in t.splitlines() if l.startswith('FLASH_UNIT')),flush=True)
 for hk in [1,4]:
  t=runner.run(f'final-head128-p{pd}-hk{hk}',[exe,pd,128,2304,1,hk,128,12,1],{'KUNLUN_FLASH_DECODING':'on','KUNLUN_FLASH_DECODING_CHUNKS':'5'},180);print(next(l for l in t.splitlines() if l.startswith('FLASH_UNIT')),flush=True)
# Private setup from the shared validation fixture, before its test loop.
ns={};exec((O/'validate.py').read_text().split('from quality import quality')[0],ns);base=ns['base']
rows=[]
for model,file in [('tiny','tinyllama-q4_0.gguf'),('7b','llama2-7b-q4_0.gguf')]:
 for mode in ['off','on']:
  name=f'audit-memory-{model}-{mode}';e={**base,'KUNLUN_FLASH_DECODING':mode,'K200_PP_LENGTHS':'2048','K200_TEACHER_FILE':str(O/'inputs/teacher64.ids'),'LD_PRELOAD':str(R/'results/flash-attention-20261026/audit.so')}
  t=runner.run(name,[R/'build/flash-decode-llama/bin/kunlun-llm-smoke',R/'models'/file,'none',8,O/(name+'.bin'),0],e,600)
  row=dict(test=name,lines=[l for l in t.splitlines() if l.startswith(('FLASH_AUDIT ','FLASH_MEMORY ','KUNLUN Flash'))]);rows.append(row);print(row,flush=True);(O/'memory-audit.json').write_text(json.dumps(rows,indent=2))
(O/'AUDIT_DONE').write_text('complete\n');print('AUDIT COMPLETE',flush=True)
