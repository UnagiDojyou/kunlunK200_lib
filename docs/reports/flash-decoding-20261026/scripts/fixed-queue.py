from pathlib import Path
import os,sys,json,time,subprocess,runpy,re,statistics,shutil,hashlib
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026';L=R/'repos/KunlunK200'
deadline=time.monotonic()+5400
while not (O/'WORK_DONE').exists():
 if (O/'XPU_ABORT').exists():raise RuntimeError('stop latch')
 if time.monotonic()>deadline:raise RuntimeError('worker comparison timeout')
 time.sleep(5)
V=O/'balanced-comparison';V.mkdir(exist_ok=True)
for build,sub in [('flash-decode-lib',''),('flash-decode-llama','kunlunk200')]:
 d=R/'build'/build/sub/'lib';shutil.copytree(d/'kunlunk200/kernels',V/build/'kernels',dirs_exist_ok=True);shutil.copy2(d/'libkunlunk200.so',V/build/'libkunlunk200.so')
runpy.run_path(str(O/'fixed-layout.py'),run_name='__main__')
p=L/'bench/flash-decode-bench.cpp';s=p.read_text();s=s.replace('auto oldrun=[&]{bool compare_work=','auto oldrun=[&]{bool compare_fixed=getenv("KUNLUN_FLASH_DECODING_COMPARE_FIXED");std::string fixed=getenv("KUNLUN_FLASH_DECODING_FIXED_KEYS")?getenv("KUNLUN_FLASH_DECODING_FIXED_KEYS"):"";if(compare_fixed)unsetenv("KUNLUN_FLASH_DECODING_FIXED_KEYS");bool compare_work=')
s=s.replace('(compare_layout||compare_work)?"on":"off"','(compare_layout||compare_work||compare_fixed)?"on":"off"').replace('if(compare_work)setenv("KUNLUN_FLASH_DECODING_WORK",work.c_str(),1);};','if(compare_work)setenv("KUNLUN_FLASH_DECODING_WORK",work.c_str(),1);if(compare_fixed)setenv("KUNLUN_FLASH_DECODING_FIXED_KEYS",fixed.c_str(),1);};');p.write_text(s)
env=os.environ.copy();env.update(XRE=str(R/'external/xre'),XDNN=str(R/'external/xdnn'),XTDK_DIR=str(R/'external/xtdk'),LD_LIBRARY_PATH=':'.join(str(R/x) for x in ['external/xre/lib64','external/xdnn/so','external/xtdk/shlib']))
for build,targets in [('flash-decode-lib',['k200-flash-decode-bench']),('flash-decode-llama',['kunlun-flash-decoding-bench','kunlun-llm-smoke','kunlun-quality-eval'])]:subprocess.run(['cmake','--build',str(R/'build'/build),'-j4','--target',*targets],env=env,check=True)
os.environ['LD_LIBRARY_PATH']=env['LD_LIBRARY_PATH'];sys.path.insert(0,str(O));import runner;runner.OUT=O
exe=R/'build/flash-decode-lib/k200-flash-decode-bench';units=[]
def test(name,pd,d,s,hk,span=1024,pattern=29,b=1,h=32,extra=None):
 e={'KUNLUN_FLASH_DECODING':'on','KUNLUN_FLASH_DECODING_FIXED_KEYS':str(span),**(extra or {})};t=runner.run(name,[exe,pd,d,s,1,hk,h,pattern,b],e,180);line=next(l for l in t.splitlines() if l.startswith('FLASH_UNIT'));print(name,line,flush=True);units.append(dict(test=name,line=line));(O/'fixed-units.json').write_text(json.dumps(units,indent=2));return t
for pd in [0,1]:
 for d,hk in [(64,4),(128,32),(128,1)]:
  for s in [2304,4352,16384]:
   for work in ['chunk','head']:test(f'fixed-p{pd}-d{d}-hk{hk}-s{s}-{work}',pd,d,s,hk,extra={'KUNLUN_FLASH_DECODING_WORK':work})
 for pattern in [3,5,6,18,24,25,27,28]:test(f'fixed-edge-p{pd}-pat{pattern}',pd,128,1056,2,512,pattern,b=4 if pattern==18 else 1,h=8,extra={'KUNLUN_FLASH_BENCH_MUTATE':'1'} if pattern in [3,18] else None)
 for s,span,c in [(256,64,4),(4096,1024,4),(16384,256,64)]:
  t=test(f'fixed-equivalent-p{pd}-s{s}',pd,128,s,4,span,12,extra={'KUNLUN_FLASH_DECODING_CHUNKS':str(c),'KUNLUN_FLASH_DECODING_COMPARE_FIXED':'1'});assert 'bitdiff=0' in t
 t=test(f'fixed-one-p{pd}',pd,128,512,4,1024,12);assert 'bitdiff=0' in t
 t=test(f'fixed-limit-p{pd}',pd,128,16384,4,32,12);assert 'Flash-Decoding' not in t and 'bitdiff=0' in t
 test(f'profile-fixed-p{pd}',pd,128,4352,32,1024,29,extra={'KUNLUN_FLASH_DECODING_WORK':'head','KUNLUN_FLASH_DECODING_PROFILE':'1'})
ns={};exec((O/'validate.py').read_text().split('from quality import quality')[0],ns);base=ns['base'];from quality import quality
results=[]
for model,file in [('7b','llama2-7b-q4_0.gguf'),('tiny','tinyllama-q4_0.gguf')]:
 allow_pd1=True
 for pd in [0,1]:
  if pd and not allow_pd1:continue
  name=f'fixed-ab-{model}-p{pd}';e={**base,'K200_CTX':'4608','K200_INPUT_IDS':str(O/'inputs/prompt4096.ids'),'K200_PP_LENGTHS':'4096','K200_TEACHER_FILE':str(O/'inputs/teacher64.ids'),'K200_REPEATS':'3','K200_FD_MODES':'off,fixed,fixhead'}
  t=runner.run(name,[R/'build/flash-decode-llama/bin/kunlun-flash-decoding-bench',R/'models'/file,'none',64,O/(name+'.bin'),pd],e,1800);print(name,'\n'+'\n'.join(l for l in t.splitlines() if l.startswith(('FDBENCH ','FDBENCH_STATE'))),flush=True)
  rates={m:[] for m in ['off','fixed','fixhead']}
  for line in t.splitlines():
   if line.startswith('FDBENCH '):f=dict(re.findall(r'(\w+)=([^ ]+)',line));rates[f['mode']].append(float(f['tps']))
  for mode in ['fixed','fixhead']:
   row=dict(test=name,model=model,pd=pd,mode=mode,tps=statistics.median(rates[mode]),gain=statistics.median(rates[mode])/statistics.median(rates['off'])-1,**quality(O/f'{name}.bin.p4096.off',O/f'{name}.bin.p4096.{mode}'));results.append(row);print(row,flush=True)
  (O/'fixed-quality.json').write_text(json.dumps(results,indent=2))
  if model=='7b' and pd==0:allow_pd1=any(statistics.median(rates[m])>statistics.median(rates['off'])*1.01 for m in ['fixed','fixhead']);print('7b PD1 follow-up',allow_pd1,flush=True)
 for_ppl=f'corpus-fixed-{model}';out=O/(for_ppl+'.bin');ref=O/f'corpus-{model}-b1-on.bin'
 t=runner.run(for_ppl,[R/'build/flash-decode-llama/bin/kunlun-quality-eval',R/'models'/file,'none',O/'inputs/corpus1024.ids',256,out,ref,'-'],{**base,'KUNLUN_FLASH_DECODING':'on','KUNLUN_FLASH_DECODING_FIXED_KEYS':'64','K200_QUALITY_BATCH':'1'},1800)
 same=hashlib.sha256(out.read_bytes()).digest()==hashlib.sha256(ref.read_bytes()).digest();assert same;results.append(dict(test=for_ppl,bitwise=same,lines=[l for l in t.splitlines() if l.startswith(('QUALITY ','CORPUS '))]));print(results[-1],flush=True);(O/'fixed-quality.json').write_text(json.dumps(results,indent=2))
(O/'FIXED_DONE').write_text('complete\n');print('FIXED COMPARISON COMPLETE',flush=True)
