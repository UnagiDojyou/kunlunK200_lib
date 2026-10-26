from pathlib import Path
import os,subprocess,sys,json,hashlib,re
R=Path('/home/unagidojyou/kunlun-k200');L=R/'repos/KunlunK200';O=R/'results/flash-online-20261026'
p=R/'repos/llama.cpp/docs/backend/kunlun-flash-online.md';s=p.read_text().replace('GGML_KUNLUN_ATTENTION=1 KUNLUN_ATTENTION_MATMUL=auto','GGML_KUNLUN_ATTENTION=1 GGML_KUNLUN_FLASH=1 KUNLUN_ATTENTION_MATMUL=auto');p.write_text(s)
p=L/'docs/flash-attention.md';s=p.read_text();addition='\n## 0.7.1 Online Softmax\n\n`KUNLUN_FLASH_ONLINE=auto` (default) selects the optimized ordered-sum CL kernel; `legacy` retains the unmodified 0.7.0 kernel. See [Online Softmax](flash-online.md) for benchmarks, numerical guarantees and the Japanese evaluation report. The FlashAttention opt-in flags above remain required.\n'
if '## 0.7.1 Online Softmax' not in s:p.write_text(s+addition)
env=os.environ.copy();env.update(XRE=str(R/'external/xre'),XDNN=str(R/'external/xdnn'),XTDK_DIR=str(R/'external/xtdk'))
subprocess.run(['cmake','--build',str(R/'build/flash-online-lib'),'-j6','--target','k200-flash-online-api-bench','k200-flash-online-bench','k200-flash-contract','k200-attention-contract','k200-common-q4-layout'],env=env,check=True)
sys.path.insert(0,str(O));import runner;runner.OUT=O
os.environ['LD_LIBRARY_PATH']=':'.join(str(R/x) for x in ['external/xre/lib64','external/xdnn/so','external/xtdk/shlib'])
for pd in [0,1]:
 for d in [64,128]:
  name=f'final-check-p{pd}-d{d}-fullq';text=runner.run(name,[R/'build/flash-online-lib/k200-flash-online-api-bench',pd,d,2048,128,8,32,14],{'KUNLUN_FLASH_ONLINE':'auto','KUNLUN_FLASH_FULL_Q_REF':'1','KUNLUN_FLASH_BENCH_SAMPLE':'1'},180);assert 'bitdiff=0' in text
  print('\n'.join(l for l in text.splitlines() if l.startswith(('FLASH_UNIT','ONLINE_COMPARE'))),flush=True)
 text=runner.run(f'final-check-standalone-p{pd}',[R/'build/flash-online-lib/k200-flash-online-bench',pd,128,512,128,32,1,1,R/'build/flash-online-lib/lib/kunlunk200/kernels'],{},60);print('\n'.join(l for l in text.splitlines() if l.startswith('ONLINE_UNIT')),flush=True)
for mode in ['legacy','auto','unknown']:
 text=runner.run(f'final-check-dispatch-{mode}',[R/'build/flash-online-lib/k200-flash-online-api-bench',0,64,256,3,2,8,12],{'KUNLUN_FLASH_ONLINE':mode},60);assert 'bitdiff=0' in text;print(mode,'dispatch pass',flush=True)
baseline=json.loads((O/'flash-lib-images.json').read_text());D=R/'build/flash-online-lib/lib/kunlunk200/kernels'
unchanged={name:hashlib.sha256((D/name).read_bytes()).hexdigest()==sha for name,sha in baseline.items()};assert all(unchanged.values())
final={'unchanged_baseline_kernels':unchanged,'states':{pd:{k:Path(f'/proc/xpu/dev{pd}/{k}').read_text().strip() for k in ['state','errinfo']} for pd in [0,1]},'reset_count':0}
assert all(x['state']=='RUNNING' and x['errinfo']=='no error' for x in final['states'].values());(O/'final-verification.json').write_text(json.dumps(final,indent=2))
print('FINAL VERIFICATION COMPLETE',flush=True)
