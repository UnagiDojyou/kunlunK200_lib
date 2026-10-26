from pathlib import Path
import os,sys,json,time,subprocess,runpy,re,statistics
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026';L=R/'repos/KunlunK200'
deadline=time.monotonic()+5400
while not (O/'AUDIT_DONE').exists():
 if (O/'XPU_ABORT').exists():raise RuntimeError('stop latch')
 if time.monotonic()>deadline:raise RuntimeError('audit timeout')
 time.sleep(5)
runpy.run_path(str(O/'worker-layout.py'),run_name='__main__')
p=L/'bench/flash-decode-bench.cpp';s=p.read_text().replace('pattern==11){qp.nb','(pattern==11||pattern==29)){qp.nb').replace('pattern==12||pattern==25){maskp.type','pattern==12||pattern==25||pattern==29){maskp.type');s=s.replace(' auto qr=q.raw,kr=k.raw,vr=v.raw,mr=mask.raw;',' if(pattern==29)for(int r=0;r<m;++r)for(int x=0;x<s;++x)mask.put(x>=s-255?-INFINITY:0.f,x,r);\n auto qr=q.raw,kr=k.raw,vr=v.raw,mr=mask.raw;');p.write_text(s)
env=os.environ.copy();env.update(XRE=str(R/'external/xre'),XDNN=str(R/'external/xdnn'),XTDK_DIR=str(R/'external/xtdk'),LD_LIBRARY_PATH=':'.join(str(R/x) for x in ['external/xre/lib64','external/xdnn/so','external/xtdk/shlib']))
for build,targets in [('flash-decode-lib',['k200-flash-decode-bench']),('flash-decode-llama',['kunlun-flash-decoding-bench','kunlun-llm-smoke','kunlun-quality-eval'])]:subprocess.run(['cmake','--build',str(R/'build'/build),'-j4','--target',*targets],env=env,check=True)
os.environ['LD_LIBRARY_PATH']=env['LD_LIBRARY_PATH'];sys.path.insert(0,str(O));import runner;runner.OUT=O
units=[];exe=R/'build/flash-decode-lib/k200-flash-decode-bench'
for pd in [0,1]:
 for d,hk in [(64,4),(128,1),(128,4),(128,32)]:
  for seq in [2048,2304,4096,4352]:
   for c in [4,5]:
    name=f'work-p{pd}-d{d}-hk{hk}-s{seq}-c{c}'
    t=runner.run(name,[exe,pd,d,seq,1,hk,32,29,1],{'KUNLUN_FLASH_DECODING':'on','KUNLUN_FLASH_DECODING_WORK':'head','KUNLUN_FLASH_DECODING_CHUNKS':str(c),'KUNLUN_FLASH_DECODING_COMPARE_WORK':'1'},180);assert 'bitdiff=0' in t and 'work=head' in t and 'work=chunk' in t
    line=next(l for l in t.splitlines() if l.startswith('FLASH_UNIT'));units.append(dict(test=name,line=line));print(name,line,flush=True);(O/'work-units.json').write_text(json.dumps(units,indent=2))
 for hk in [1,4,32]:
  name=f'profile-work-p{pd}-hk{hk}'
  t=runner.run(name,[exe,pd,128,4352,1,hk,32,29,1],{'KUNLUN_FLASH_DECODING':'on','KUNLUN_FLASH_DECODING_WORK':'head','KUNLUN_FLASH_DECODING_CHUNKS':'5','KUNLUN_FLASH_DECODING_PROFILE':'1'},180);print(name,'pass',flush=True)
ns={};exec((O/'validate.py').read_text().split('from quality import quality')[0],ns);base=ns['base'];from quality import quality
results=[]
for model,file in [('7b','llama2-7b-q4_0.gguf'),('tiny','tinyllama-q4_0.gguf')]:
 for pd in [0,1]:
  name=f'work-ab-{model}-p{pd}';e={**base,'K200_CTX':'4608','K200_INPUT_IDS':str(O/'inputs/prompt4096.ids'),'K200_PP_LENGTHS':'4096','K200_TEACHER_FILE':str(O/'inputs/teacher64.ids'),'K200_REPEATS':'3','K200_FD_MODES':'off,on:5,head:5'}
  t=runner.run(name,[R/'build/flash-decode-llama/bin/kunlun-flash-decoding-bench',R/'models'/file,'none',64,O/(name+'.bin'),pd],e,1800);print(name,'\n'+'\n'.join(l for l in t.splitlines() if l.startswith(('FDBENCH ','FDBENCH_STATE'))),flush=True)
  a=O/f'{name}.bin.p4096.on:5';b=O/f'{name}.bin.p4096.head:5';same=a.read_bytes()==b.read_bytes();assert same
  results.append(dict(test=name,work_bitwise=same,**quality(O/f'{name}.bin.p4096.off',b)));print(results[-1],flush=True);(O/'work-quality.json').write_text(json.dumps(results,indent=2))
(O/'WORK_DONE').write_text('complete\n');print('WORK COMPARISON COMPLETE',flush=True)
