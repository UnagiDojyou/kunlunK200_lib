from pathlib import Path
import os,sys,json,re,hashlib
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026';sys.path.insert(0,str(O));import runner;runner.OUT=O
os.environ['LD_LIBRARY_PATH']=':'.join(str(R/x) for x in ['external/xre/lib64','external/xdnn/so','external/xtdk/shlib'])
exe=R/'build/flash-decode-lib/k200-flash-decode-bench';records=[]
def unit(name,pd,d,s,hk,m=1,pat=29,env=None,active=False):
 t=runner.run(name,[exe,pd,d,s,m,hk,32,pat,1],{'KUNLUN_FLASH_DECODING':'auto',**(env or {})},180)
 assert ('Flash-Decoding' in t)==active
 if not active:assert 'bitdiff=0' in t
 line=next(l for l in t.splitlines() if l.startswith('FLASH_UNIT'));records.append(dict(test=name,active=active,line=line));print(name,line,flush=True);(O/'final-tests.json').write_text(json.dumps(records,indent=2));return t
for pd in [0,1]:
 for s in [2048,2304,2560,2816,3072,3328,3584,3840,4096,4352]:unit(f'auto-unit-p{pd}-s{s}',pd,64,s,4,active=True)
 for d,s,hk in [(64,1280,4),(64,8192,4),(128,4352,32)]:unit(f'fallback-auto-p{pd}-d{d}-s{s}-hk{hk}',pd,d,s,hk)
 for name,e in [('kq',{'KUNLUN_FLASH_KQ':'legacy'}),('unknown-kq',{'KUNLUN_FLASH_KQ':'unknown'}),('online',{'KUNLUN_FLASH_ONLINE':'legacy'}),('pv',{'KUNLUN_FLASH_PV':'narrow'}),('unknown',{'KUNLUN_FLASH_DECODING':'unknown'}),('hbm',{'KUNLUN_FLASH_HBM':'1'})]:unit(f'fallback-{name}-p{pd}',pd,64,4352,4,env=e)
 unit(f'fallback-pp-p{pd}',pd,64,512,4,m=128,pat=12)
 for d,hk in [(64,4),(128,32)]:
  for pat in [0,11]:
   # Same logical input and F32 mask; only K/V physical layout differs.
   name=f'layout-control-p{pd}-d{d}-pat{pat}'
   e={'KUNLUN_FLASH_DECODING':'on','KUNLUN_FLASH_DECODING_FIXED_KEYS':'1024','KUNLUN_FLASH_DECODING_WORK':'head'}
   unit(name,pd,d,4352,hk,pat=pat,env=e,active=True)
   t=runner.run('profile-'+name,[exe,pd,d,4352,1,hk,32,pat,1],{**e,'KUNLUN_FLASH_DECODING_PROFILE':'1'},180);print('profile-'+name,'pass',flush=True)
ns={};exec((O/'validate.py').read_text().split('from quality import quality')[0],ns);base=ns['base'];from quality import quality
for pd in [0,1]:
 name=f'final-auto-tiny-p{pd}';e={**base,'K200_CTX':'4608','K200_INPUT_IDS':str(O/'inputs/prompt4096.ids'),'K200_PP_LENGTHS':'4096','K200_TEACHER_FILE':str(O/'inputs/teacher64.ids'),'K200_REPEATS':'2','K200_FD_MODES':'off,auto'}
 t=runner.run(name,[R/'build/flash-decode-llama/bin/kunlun-flash-decoding-bench',R/'models/tinyllama-q4_0.gguf','none',64,O/(name+'.bin'),pd],e,1200)
 assert 'fixed=1024' in t and 'work=head' in t
 a=O/f'{name}.bin.p4096.auto';b=O/f'fixed-ab-tiny-p{pd}.bin.p4096.fixhead';assert a.read_bytes()==b.read_bytes()
 row=dict(test=name,measured_method_bitwise=True,lines=[l for l in t.splitlines() if l.startswith(('FDBENCH ','FDBENCH_STATE'))],**quality(O/f'{name}.bin.p4096.off',a));records.append(row);print(row,flush=True);(O/'final-tests.json').write_text(json.dumps(records,indent=2))
for model,file in [('tiny','tinyllama-q4_0.gguf'),('7b','llama2-7b-q4_0.gguf')]:
 for split in ['layer','row']:
  for mode in ['off','auto']:
   name=f'final-split-{model}-{split}-{mode}';e={**base,'K200_PP_LENGTHS':'128,512,2048','K200_TEACHER_FILE':str(O/'inputs/teacher64.ids'),'KUNLUN_FLASH_DECODING':mode}
   t=runner.run(name,[R/'build/flash-decode-llama/bin/kunlun-llm-smoke',R/'models'/file,split,16,O/(name+'.bin'),0],e,1200)
   records.append(dict(test=name,perf=[l for l in t.splitlines() if l.startswith('K200PERF')]));print(records[-1],flush=True);(O/'final-tests.json').write_text(json.dumps(records,indent=2))
  for n in [128,512,2048]:
   a=O/f'final-split-{model}-{split}-off.bin.p{n}';b=O/f'final-split-{model}-{split}-auto.bin.p{n}';row=dict(test=f'final-split-quality-{model}-{split}-p{n}',bitwise=a.read_bytes()==b.read_bytes(),**quality(a,b));records.append(row);print(row,flush=True)
   if model=='7b' or n<2048:assert row['bitwise']
   (O/'final-tests.json').write_text(json.dumps(records,indent=2))
# Memory/launch/host-copy audit is separate from all normal performance runs.
for model,file,n,ctx in [('tiny','tinyllama-q4_0.gguf',4096,4608),('7b','llama2-7b-q4_0.gguf',2048,4096)]:
 for mode in ['off','auto']:
  name=f'final-memory-{model}-{mode}';e={**base,'K200_CTX':str(ctx),'K200_INPUT_IDS':str(O/'inputs/prompt4096.ids' if n==4096 else O/'inputs/prompt2048.ids'),'K200_PP_LENGTHS':str(n),'K200_TEACHER_FILE':str(O/'inputs/teacher64.ids'),'KUNLUN_FLASH_DECODING':mode,'LD_PRELOAD':str(R/'results/flash-attention-20261026/audit.so')}
  t=runner.run(name,[R/'build/flash-decode-llama/bin/kunlun-llm-smoke',R/'models'/file,'none',8,O/(name+'.bin'),0],e,1200);row=dict(test=name,lines=[l for l in t.splitlines() if l.startswith(('FLASH_AUDIT ','FLASH_MEMORY ','KUNLUN Flash'))]);records.append(row);print(name,'pass',flush=True);(O/'final-tests.json').write_text(json.dumps(records,indent=2))
for pd in [0,1]:
 for target in ['kunlun-attention-test','kunlun-swiglu-test']:
  runner.run(f'final-regression-{target}-p{pd}',[R/'build/flash-decode-llama/bin'/target,pd],{'GGML_KUNLUN_ATTENTION':'1','GGML_KUNLUN_SWIGLU':'1','GGML_KUNLUN_ROPE':'1','KUNLUN_ATTENTION_MATMUL':'cl','KUNLUN_FLASH_KV':'1'},180)
for target in ['k200-flash-contract','k200-attention-contract','k200-common-q4-layout']:runner.run('final-regression-'+target,[R/'build/flash-decode-lib'/target],{},120)
(O/'FINAL_TESTS_DONE').write_text('complete\n');print('FINAL TESTS COMPLETE',flush=True)
