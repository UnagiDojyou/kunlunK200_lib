from pathlib import Path
import os,sys,json
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-online-20261026';sys.path.insert(0,str(O));import runner;runner.OUT=O
os.environ['LD_LIBRARY_PATH']=':'.join(str(R/x) for x in ['external/xre/lib64','external/xdnn/so','external/xtdk/shlib'])
base={'OMP_WAIT_POLICY':'PASSIVE','K200_CTX':'4096','K200_BATCH':'128','K200_UBATCH':'128','K200_INPUT_IDS':str(O/'inputs/prompt2048.ids'),'K200_TEACHER_FILE':str(O/'inputs/teacher64.ids'),'K200_PP_LENGTHS':'128,512,2048','K200_WARMUP':'1','K200_REPEATS':'2','K200_IGNORE_EOS':'1','GGML_KUNLUN_COMMON_Q4':'1','KUNLUN_BROADCAST_VECTOR':'1','GGML_KUNLUN_ROPE':'1','GGML_KUNLUN_SWIGLU':'0','GGML_KUNLUN_ATTENTION':'1','KUNLUN_ATTENTION_MATMUL':'auto','KUNLUN_ATTENTION_SOFTMAX':'fast','KUNLUN_ATTENTION_PREP':'auto','KUNLUN_FLASH_KV':'1','K200_FLASH':'1','GGML_KUNLUN_FLASH':'1'}
results=[]
from quality import quality
for model,file in [('tiny','tinyllama-q4_0.gguf'),('7b','llama2-7b-q4_0.gguf')]:
 for pd in [0,1]:
  for mode in (['legacy','ordered'] if pd==0 else ['ordered','legacy']):
   name=f'final-perf-{model}-p{pd}-{mode}';env={**base,'KUNLUN_FLASH_ONLINE':mode}
   t=runner.run(name,[R/'build/flash-online-llama/bin/kunlun-llm-smoke',R/'models'/file,'none',16,O/(name+'.bin'),pd],env,1200)
   print(name,'\n'+'\n'.join(l for l in t.splitlines() if l.startswith('K200PERF')),flush=True)
  for n in [128,512,2048]:
   row=dict(test=f'{model}-p{pd}-p{n}',**quality(O/f'final-perf-{model}-p{pd}-legacy.bin.p{n}',O/f'final-perf-{model}-p{pd}-ordered.bin.p{n}'));results.append(row);print(row,flush=True)
  (O/'logits-quality.json').write_text(json.dumps(results,indent=2))
print('PERFORMANCE COMPLETE',flush=True)

