from pathlib import Path
import os,sys,json,hashlib
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026';sys.path.insert(0,str(O));import runner;runner.OUT=O
os.environ['LD_LIBRARY_PATH']=':'.join(str(R/x) for x in ['external/xre/lib64','external/xdnn/so','external/xtdk/shlib'])
base={'KUNLUN_FLASH_ONLINE':'auto','KUNLUN_FLASH_KQ':'auto','OMP_WAIT_POLICY':'PASSIVE','K200_CTX':'4096','K200_BATCH':'128','K200_UBATCH':'128','K200_INPUT_IDS':str(O/'inputs/prompt2048.ids'),'K200_WARMUP':'0','K200_REPEATS':'1','K200_IGNORE_EOS':'1','GGML_KUNLUN_COMMON_Q4':'1','KUNLUN_BROADCAST_VECTOR':'1','GGML_KUNLUN_ROPE':'1','GGML_KUNLUN_SWIGLU':'0','GGML_KUNLUN_ATTENTION':'1','KUNLUN_ATTENTION_MATMUL':'auto','KUNLUN_ATTENTION_SOFTMAX':'fast','KUNLUN_ATTENTION_PREP':'auto','KUNLUN_FLASH_KV':'1','K200_FLASH':'1','GGML_KUNLUN_FLASH':'1'}
from quality import quality
results=[]
def save(row):
 results.append(row);print(row,flush=True);(O/'validation.json').write_text(json.dumps(results,indent=2))
for model,file in [('tiny','tinyllama-q4_0.gguf'),('7b','llama2-7b-q4_0.gguf')]:
 for batch,mode in [(128,'off'),(128,'on'),(1,'off'),(1,'on')]:
  name=f'corpus-{model}-b{batch}-{mode}';out=O/(name+'.bin');ref=(O/f'corpus-{model}-b1-off.bin') if batch==1 and mode=='on' else R/f'results/flash-kq-20261026/corpus-{model}-kq.bin'
  env={**base,'KUNLUN_FLASH_DECODING':mode,'KUNLUN_FLASH_DECODING_CHUNKS':'4','K200_QUALITY_BATCH':str(batch)}
  t=runner.run(name,[R/'build/flash-decode-llama/bin/kunlun-quality-eval',R/'models'/file,'none',O/'inputs/corpus1024.ids',256,out,ref,'-'],env,1800)
  row=dict(test=name,lines=[l for l in t.splitlines() if l.startswith(('QUALITY ','CORPUS '))],sha256=hashlib.sha256(out.read_bytes()).hexdigest())
  if batch==128:row['bitwise']=out.read_bytes()==ref.read_bytes();assert row['bitwise']
  save(row)
 for split in ['layer','row']:
  for mode in ['off','on']:
   name=f'compat-{model}-{split}-{mode}';env={**base,'K200_PP_LENGTHS':'128,2048','K200_TEACHER_FILE':str(O/'inputs/teacher64.ids'),'KUNLUN_FLASH_DECODING':mode,'K200_CHECK_STATE':'1'}
   t=runner.run(name,[R/'build/flash-decode-llama/bin/kunlun-llm-smoke',R/'models'/file,split,32,O/(name+'.bin'),0],env,1800);assert t.count('restored_next_logits_bitwise=1')==2
   save(dict(test=name,state_restore=True,perf=[l for l in t.splitlines() if l.startswith('K200PERF')]))
  for n in [128,2048]:save(dict(test=f'quality-{model}-{split}-p{n}',**quality(O/f'compat-{model}-{split}-off.bin.p{n}',O/f'compat-{model}-{split}-on.bin.p{n}')))
 for n,steps in [(128,32),(2048,64)]:
  for mode in ['off','on']:
   name=f'greedy-{model}-p{n}-{mode}';env={**base,'K200_PP_LENGTHS':str(n),'KUNLUN_FLASH_DECODING':mode,'K200_CHECK_STATE':'1'}
   t=runner.run(name,[R/'build/flash-decode-llama/bin/kunlun-llm-smoke',R/'models'/file,'none',steps,O/(name+'.bin'),0],env,1800);assert 'restored_next_logits_bitwise=1' in t
   save(dict(test=name,state_restore=True,tokens=[l for l in t.splitlines() if l.startswith('TOKENS ')]))
  save(dict(test=f'greedy-quality-{model}-p{n}',**quality(O/f'greedy-{model}-p{n}-off.bin',O/f'greedy-{model}-p{n}-on.bin')))
for pd in [0,1]:
 for target in ['kunlun-attention-test','kunlun-swiglu-test']:
  t=runner.run(f'regression-{target}-p{pd}',[R/'build/flash-decode-llama/bin'/target,pd],{'KUNLUN_FLASH_KV':'1','GGML_KUNLUN_ATTENTION':'1','GGML_KUNLUN_SWIGLU':'1','GGML_KUNLUN_ROPE':'1','KUNLUN_ATTENTION_MATMUL':'cl'},180);save(dict(test=f'{target}-p{pd}',pass_result=True))
for target in ['k200-flash-contract','k200-attention-contract','k200-common-q4-layout']:
 runner.run('regression-'+target,[R/'build/flash-decode-lib'/target],{},120);save(dict(test=target,pass_result=True))
print('VALIDATION COMPLETE',flush=True)
