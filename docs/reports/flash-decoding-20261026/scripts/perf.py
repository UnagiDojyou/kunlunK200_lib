from pathlib import Path
import os,sys,json,runpy
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026';sys.path.insert(0,str(O));import runner;runner.OUT=O
os.environ['LD_LIBRARY_PATH']=':'.join(str(R/x) for x in ['external/xre/lib64','external/xdnn/so','external/xtdk/shlib'])
base={'KUNLUN_FLASH_ONLINE':'auto','KUNLUN_FLASH_KQ':'auto','OMP_WAIT_POLICY':'PASSIVE','K200_CTX':'4096','K200_BATCH':'128','K200_UBATCH':'128','K200_INPUT_IDS':str(O/'inputs/prompt2048.ids'),'K200_TEACHER_FILE':str(O/'inputs/teacher64.ids'),'K200_PP_LENGTHS':'128,512,1024,2048','K200_WARMUP':'1','K200_REPEATS':'2','K200_IGNORE_EOS':'1','GGML_KUNLUN_COMMON_Q4':'1','KUNLUN_BROADCAST_VECTOR':'1','GGML_KUNLUN_ROPE':'1','GGML_KUNLUN_SWIGLU':'0','GGML_KUNLUN_ATTENTION':'1','KUNLUN_ATTENTION_MATMUL':'auto','KUNLUN_ATTENTION_SOFTMAX':'fast','KUNLUN_ATTENTION_PREP':'auto','KUNLUN_FLASH_KV':'1','K200_FLASH':'1','GGML_KUNLUN_FLASH':'1'}
for key in list(os.environ):
 if any(x in key for x in ['TRACE','PROFILE','LD_PRELOAD','CHECK_STATE']):os.environ.pop(key,None)
from quality import quality
results=[]
for model,file in [('tiny','tinyllama-q4_0.gguf'),('7b','llama2-7b-q4_0.gguf')]:
 for pd in [0,1]:
  for mode in (['off','on'] if pd==0 else ['on','off']):
   name=f'perf-{model}-p{pd}-{mode}'
   t=runner.run(name,[R/'build/flash-decode-llama/bin/kunlun-llm-smoke',R/'models'/file,'none',64,O/(name+'.bin'),pd],{**base,'KUNLUN_FLASH_DECODING':mode},1800)
   print(name,'\n'+'\n'.join(l for l in t.splitlines() if l.startswith('K200PERF')),flush=True)
  for n in [128,512,1024,2048]:
   row=dict(model=model,pd=pd,context=n,**quality(O/f'perf-{model}-p{pd}-off.bin.p{n}',O/f'perf-{model}-p{pd}-on.bin.p{n}'));results.append(row);print(row,flush=True)
  (O/'perf-quality.json').write_text(json.dumps(results,indent=2))
print('NORMAL PERFORMANCE COMPLETE',flush=True)
for model,file in [('tiny','tinyllama-q4_0.gguf'),('7b','llama2-7b-q4_0.gguf')]:
 for mode in ['off','on']:
  name=f'long-perf-{model}-{mode}';env={**base,'KUNLUN_FLASH_DECODING':mode,'K200_CTX':'4608','K200_PP_LENGTHS':'4096','K200_INPUT_IDS':str(O/'inputs/prompt4096.ids')}
  t=runner.run(name,[R/'build/flash-decode-llama/bin/kunlun-llm-smoke',R/'models'/file,'none',64,O/(name+'.bin'),0],env,1800);print(name,'\n'+'\n'.join(l for l in t.splitlines() if l.startswith('K200PERF')),flush=True)
 row=dict(model=model,pd=0,context=4096,**quality(O/f'long-perf-{model}-off.bin',O/f'long-perf-{model}-on.bin'));results.append(row);print(row,flush=True);(O/'perf-quality.json').write_text(json.dumps(results,indent=2))
print('LONG PERFORMANCE COMPLETE',flush=True)
