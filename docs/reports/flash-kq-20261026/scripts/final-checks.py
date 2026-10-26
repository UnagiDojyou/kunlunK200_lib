from pathlib import Path
import os,sys,subprocess,json,hashlib,re
R=Path('/home/unagidojyou/kunlun-k200');L=R/'repos/KunlunK200';O=R/'results/flash-kq-20261026';sys.path.insert(0,str(O));import runner;runner.OUT=O
os.environ['LD_LIBRARY_PATH']=':'.join(str(R/x) for x in ['external/xre/lib64','external/xdnn/so','external/xtdk/shlib'])
for pd in [0,1]:
 for d in [64,128]:
  name=f'final-check-p{pd}-d{d}-fullq';text=runner.run(name,[R/'build/flash-kq-lib/k200-flash-kq-api-bench',pd,d,2048,128,8,32,14],{'KUNLUN_FLASH_KQ':'auto','KUNLUN_FLASH_ONLINE':'auto','KUNLUN_FLASH_FULL_Q_REF':'1','KUNLUN_FLASH_BENCH_SAMPLE':'1'},180);assert 'bitdiff=0' in text
  print('\n'.join(l for l in text.splitlines() if l.startswith(('FLASH_UNIT','KQ_COMPARE'))),flush=True)
 for kind,args,extra in [('tg-ungrouped',[128,2048,1,8,32,17],{'KUNLUN_FLASH_TG_GROUP':'0'}),('tg-one-kv',[128,2048,1,1,32,17],{}),('wide-block',[128,2048,64,2,8,12],{'KUNLUN_FLASH_BLOCK':'1024'}),('query16',[128,512,128,8,32,12],{'KUNLUN_FLASH_QUERY_TILE':'16'}),('llama-pp',[128,2048,128,32,32,12],{}),('llama-tg',[128,2048,1,32,32,12],{}),('three-kv-heads',[128,544,17,3,6,13],{})]:
  text=runner.run(f'final-check-{kind}-p{pd}',[R/'build/flash-kq-lib/k200-flash-kq-api-bench',pd,*args],{'KUNLUN_FLASH_KQ':'auto','KUNLUN_FLASH_ONLINE':'auto','KUNLUN_FLASH_BENCH_SAMPLE':'1',**extra},180);assert 'bitdiff=0' in text;print('\n'.join(l for l in text.splitlines() if l.startswith(('FLASH_UNIT','KQ_COMPARE'))),flush=True)
 text=runner.run(f'final-standalone-p{pd}',[R/'build/flash-kq-lib/k200-flash-kq-bench',pd,128,2048,128,32,32,R/'build/flash-kq-lib/lib/kunlunk200/kernels'],{},120);print('\n'.join(l for l in text.splitlines() if l.startswith('KQ_UNIT')),flush=True)
for mode,s,hk,expected in [('auto',32,4,'legacy'),('auto',128,1,'legacy'),('auto',128,4,'reuse1024'),('auto',512,4,'reuse1024'),('legacy',512,4,'legacy'),('unknown',512,4,'legacy'),('reuse1024',32,4,'reuse1024')]:
 name=f'final-check-dispatch-{mode}-s{s}-hk{hk}';text=runner.run(name,[R/'build/flash-kq-lib/k200-flash-kq-api-bench',0,64,s,3,hk,8,12],{'KUNLUN_FLASH_KQ':mode,'KUNLUN_FLASH_ONLINE':'auto'},120)
 first=next(l for l in text.splitlines() if l.startswith('KUNLUN Flash '));assert 'kq='+expected in first and 'bitdiff=0' in text;print(name,first,flush=True)
# Recheck automatic routing against the exact 0.7.1 corpus, after host cleanup/version bump.
base={'KUNLUN_FLASH_KQ':'auto','KUNLUN_FLASH_ONLINE':'auto','OMP_WAIT_POLICY':'PASSIVE','GGML_KUNLUN_COMMON_Q4':'1','KUNLUN_BROADCAST_VECTOR':'1','GGML_KUNLUN_ROPE':'1','GGML_KUNLUN_SWIGLU':'0','GGML_KUNLUN_ATTENTION':'1','KUNLUN_ATTENTION_MATMUL':'auto','KUNLUN_ATTENTION_SOFTMAX':'fast','KUNLUN_ATTENTION_PREP':'auto','KUNLUN_FLASH_KV':'1','K200_QUALITY_BATCH':'128','K200_FLASH':'1','GGML_KUNLUN_FLASH':'1'}
quality=[]
for model,file in [('tiny','tinyllama-q4_0.gguf'),('7b','llama2-7b-q4_0.gguf')]:
 out=O/f'corpus-final-{model}.bin';ref=R/f'results/flash-online-20261026/corpus-{model}-ordered.bin'
 text=runner.run('corpus-final-'+model,[R/'build/flash-kq-llama/bin/kunlun-quality-eval',R/'models'/file,'none',O/'inputs/corpus1024.ids',256,out,ref],base,1200)
 same=out.read_bytes()==ref.read_bytes();assert same;quality.append(dict(model=model,bitwise=same,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),lines=[l for l in text.splitlines() if l.startswith(('QUALITY ','CORPUS '))]));print(quality[-1],flush=True)
(O/'final-corpus-quality.json').write_text(json.dumps(quality,indent=2))
states={p:{k:Path(f'/proc/xpu/dev{p}/{k}').read_text().strip() for k in ['state','errinfo']} for p in [0,1]};assert all(x['state']=='RUNNING' and x['errinfo']=='no error' for x in states.values())
base=json.loads((O/'flash-online-lib-images.json').read_text());D=R/'build/flash-kq-lib/lib/kunlunk200/kernels';same={n:hashlib.sha256((D/n).read_bytes()).hexdigest()==h for n,h in base.items() if (D/n).exists()};assert len(same)==51 and all(same.values())
for repo in [L,R/'repos/llama.cpp']:subprocess.run(['git','diff','--check'],cwd=repo,check=True)
(O/'final-verification.json').write_text(json.dumps(dict(states=states,reset_count=0,unchanged_baseline_kernels=same),indent=2));print('FINAL VERIFICATION COMPLETE',flush=True)
