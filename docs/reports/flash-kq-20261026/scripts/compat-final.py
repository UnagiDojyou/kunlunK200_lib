from pathlib import Path
import os,sys,json,hashlib
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-kq-20261026';sys.path.insert(0,str(O));import runner;runner.OUT=O
os.environ['LD_LIBRARY_PATH']=':'.join(str(R/x) for x in ['external/xre/lib64','external/xdnn/so','external/xtdk/shlib'])
base={'KUNLUN_FLASH_ONLINE':'auto','OMP_WAIT_POLICY':'PASSIVE','K200_CTX':'4096','K200_BATCH':'128','K200_UBATCH':'128','K200_INPUT_IDS':str(O/'inputs/prompt2048.ids'),'K200_IGNORE_EOS':'1','GGML_KUNLUN_COMMON_Q4':'1','KUNLUN_BROADCAST_VECTOR':'1','GGML_KUNLUN_ROPE':'1','GGML_KUNLUN_SWIGLU':'0','GGML_KUNLUN_ATTENTION':'1','KUNLUN_ATTENTION_MATMUL':'auto','KUNLUN_ATTENTION_SOFTMAX':'fast','KUNLUN_ATTENTION_PREP':'auto','KUNLUN_FLASH_KV':'1','K200_FLASH':'1','GGML_KUNLUN_FLASH':'1','K200_WARMUP':'0','K200_REPEATS':'1','K200_CHECK_STATE':'1'}
from quality import quality
results=[]
for model,file in [('tiny','tinyllama-q4_0.gguf'),('7b','llama2-7b-q4_0.gguf')]:
 for split in ['none','layer','row']:
  for mode in ['legacy','reuse1024']:
   env={**base,'K200_PP_LENGTHS':'128,512','K200_TEACHER_FILE':str(O/'inputs/teacher64.ids'),'KUNLUN_FLASH_KQ':mode}
   name=f'compat-{model}-{split}-{mode}';text=runner.run(name,[R/'build/flash-kq-llama/bin/kunlun-llm-smoke',R/'models'/file,split,32,O/(name+'.bin'),0],env,1200);assert 'restored_next_logits_bitwise=1' in text
   print(name,'\n'+'\n'.join(l for l in text.splitlines() if l.startswith(('K200STATE','K200PERF'))),flush=True)
  for n in [128,512]:
   a=O/f'compat-{model}-{split}-legacy.bin.p{n}';b=O/f'compat-{model}-{split}-reuse1024.bin.p{n}'
   row=dict(test=f'{model}-{split}-p{n}',bitwise=a.read_bytes()==b.read_bytes(),**quality(a,b));results.append(row);print(row,flush=True);assert row['bitwise']
 for n,steps in [(128,32),(2048,64)]:
  for mode in ['legacy','reuse1024']:
   name=f'greedy-{model}-p{n}-{mode}';env={**base,'K200_PP_LENGTHS':str(n),'KUNLUN_FLASH_KQ':mode}
   text=runner.run(name,[R/'build/flash-kq-llama/bin/kunlun-llm-smoke',R/'models'/file,'none',steps,O/(name+'.bin'),0],env,1200);assert 'restored_next_logits_bitwise=1' in text;print(name,'state restore pass',flush=True)
  a=O/f'greedy-{model}-p{n}-legacy.bin';b=O/f'greedy-{model}-p{n}-reuse1024.bin';same=a.read_bytes()==b.read_bytes();results.append(dict(test=f'greedy-{model}-p{n}',bitwise=same));print(results[-1],flush=True);assert same
 (O/'compat-final.json').write_text(json.dumps(results,indent=2))
for pd in [0,1]:
 for target in ['kunlun-attention-test','kunlun-swiglu-test']:
  env={'KUNLUN_FLASH_KV':'1','GGML_KUNLUN_ATTENTION':'1','GGML_KUNLUN_SWIGLU':'1','GGML_KUNLUN_ROPE':'1','KUNLUN_ATTENTION_MATMUL':'cl'}
  t=runner.run(f'regression-{target}-p{pd}',[R/'build/flash-kq-llama/bin'/target,pd],env,180);print(target,pd,'pass',flush=True)
for target in ['k200-flash-contract','k200-attention-contract','k200-common-q4-layout']:
 t=runner.run('regression-'+target,[R/'build/flash-kq-lib'/target],{},120);print(target,'pass',flush=True)
print('COMPATIBILITY COMPLETE',flush=True)
