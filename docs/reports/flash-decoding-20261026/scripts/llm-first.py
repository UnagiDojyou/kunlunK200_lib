from pathlib import Path
import os,sys,json,math,array
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026';sys.path.insert(0,str(O));import runner;runner.OUT=O
os.environ['LD_LIBRARY_PATH']=':'.join(str(R/x) for x in ['external/xre/lib64','external/xdnn/so','external/xtdk/shlib'])
base={'KUNLUN_FLASH_ONLINE':'auto','KUNLUN_FLASH_KQ':'auto','OMP_WAIT_POLICY':'PASSIVE','K200_CTX':'4096','K200_BATCH':'128','K200_UBATCH':'128','K200_INPUT_IDS':str(O/'inputs/prompt2048.ids'),'K200_TEACHER_FILE':str(O/'inputs/teacher64.ids'),'K200_PP_LENGTHS':'2048','K200_WARMUP':'0','K200_REPEATS':'1','K200_IGNORE_EOS':'1','GGML_KUNLUN_COMMON_Q4':'1','KUNLUN_BROADCAST_VECTOR':'1','GGML_KUNLUN_ROPE':'1','GGML_KUNLUN_SWIGLU':'0','GGML_KUNLUN_ATTENTION':'1','KUNLUN_ATTENTION_MATMUL':'auto','KUNLUN_ATTENTION_SOFTMAX':'fast','KUNLUN_ATTENTION_PREP':'auto','KUNLUN_FLASH_KV':'1','K200_FLASH':'1','GGML_KUNLUN_FLASH':'1'}
from quality import quality
rows=[]
for model,file in [('tiny','tinyllama-q4_0.gguf'),('7b','llama2-7b-q4_0.gguf')]:
 for mode,c in [('off',None),('on',2),('on',4)]:
  name=f'first-llm-{model}-{mode}'+(f'-c{c}' if c else '');env={**base,'KUNLUN_FLASH_DECODING':mode}
  if c:env['KUNLUN_FLASH_DECODING_CHUNKS']=str(c)
  t=runner.run(name,[R/'build/flash-decode-llama/bin/kunlun-llm-smoke',R/'models'/file,'none',32,O/(name+'.bin'),0],env,1200);print(name,'\n'+'\n'.join(l for l in t.splitlines() if l.startswith('K200PERF')),flush=True)
  if c:
   row=dict(model=model,chunks=c,**quality(O/f'first-llm-{model}-off.bin',O/(name+'.bin')));rows.append(row);print(row,flush=True);(O/'first-llm-quality.json').write_text(json.dumps(rows,indent=2))
print('FIRST LLM COMPLETE',flush=True)
