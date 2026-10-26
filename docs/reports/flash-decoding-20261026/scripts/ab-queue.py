from pathlib import Path
import os,sys,time,json,runpy,subprocess,re,statistics
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026'
deadline=time.monotonic()+5400
while not (O/'VALIDATION_DONE').exists():
 if (O/'XPU_ABORT').exists():raise RuntimeError('stop latch')
 if time.monotonic()>deadline:raise RuntimeError('validation timeout')
 time.sleep(5)
runpy.run_path(str(O/'add-ab-bench.py'),run_name='__main__')
env=os.environ.copy();env.update(XRE=str(R/'external/xre'),XDNN=str(R/'external/xdnn'),XTDK_DIR=str(R/'external/xtdk'),LD_LIBRARY_PATH=':'.join(str(R/x) for x in ['external/xre/lib64','external/xdnn/so','external/xtdk/shlib']))
subprocess.run(['cmake','--build',str(R/'build/flash-decode-llama'),'-j4','--target','kunlun-flash-decoding-bench','kunlun-llm-smoke','kunlun-quality-eval'],env=env,check=True)
os.environ['LD_LIBRARY_PATH']=env['LD_LIBRARY_PATH'];sys.path.insert(0,str(O));import runner;runner.OUT=O
base={'KUNLUN_FLASH_ONLINE':'auto','KUNLUN_FLASH_KQ':'auto','OMP_WAIT_POLICY':'PASSIVE','K200_CTX':'4608','K200_BATCH':'128','K200_UBATCH':'128','K200_INPUT_IDS':str(O/'inputs/prompt4096.ids'),'K200_TEACHER_FILE':str(O/'inputs/teacher64.ids'),'K200_PP_LENGTHS':'4096','K200_REPEATS':'3','K200_IGNORE_EOS':'1','GGML_KUNLUN_COMMON_Q4':'1','KUNLUN_BROADCAST_VECTOR':'1','GGML_KUNLUN_ROPE':'1','GGML_KUNLUN_SWIGLU':'0','GGML_KUNLUN_ATTENTION':'1','KUNLUN_ATTENTION_MATMUL':'auto','KUNLUN_ATTENTION_SOFTMAX':'fast','KUNLUN_ATTENTION_PREP':'auto','KUNLUN_FLASH_KV':'1','K200_FLASH':'1','GGML_KUNLUN_FLASH':'1'}
results=[]
from quality import quality
for model,file,d,hk in [('tiny','tinyllama-q4_0.gguf',64,4),('7b','llama2-7b-q4_0.gguf',128,32)]:
 for pd in [0,1]:
  choices=[]
  for c in [2,3,4,5,8]:
   p=O/f'logs/padded-p{pd}-d{d}-hk{hk}-s4352-c{c}.txt';s=p.read_text();line=next(l for l in s.splitlines() if l.startswith('FLASH_UNIT'));ms=float(re.search(r'flash_ms=([^ ]+)',line)[1]);choices.append((ms,c))
  best=min(choices)[1];modes='off,on'+(f',on:{best}' if best!=8 else '')
  name=f'ab-{model}-p{pd}';t=runner.run(name,[R/'build/flash-decode-llama/bin/kunlun-flash-decoding-bench',R/'models'/file,'none',64,O/(name+'.bin'),pd],{**base,'K200_FD_MODES':modes},1800)
  lines=[l for l in t.splitlines() if l.startswith(('FDBENCH ','FDBENCH_STATE'))];print(name,'\n'+'\n'.join(lines),flush=True)
  for mode in modes.split(',')[1:]:
   row=dict(test=name,model=model,pd=pd,mode=mode,unit_best=best,**quality(O/f'{name}.bin.p4096.off',O/f'{name}.bin.p4096.{mode}'));results.append(row);print(row,flush=True)
  (O/'ab-quality.json').write_text(json.dumps(results,indent=2))
(O/'AB_DONE').write_text('complete\n');print('AB COMPLETE',flush=True)
