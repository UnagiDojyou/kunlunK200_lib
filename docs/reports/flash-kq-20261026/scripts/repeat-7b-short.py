from pathlib import Path
import runpy,os,sys,json
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-kq-20261026';sys.path.insert(0,str(O));import runner;runner.OUT=O
os.environ['LD_LIBRARY_PATH']=':'.join(str(R/x) for x in ['external/xre/lib64','external/xdnn/so','external/xtdk/shlib'])
base={'KUNLUN_FLASH_ONLINE':'auto','OMP_WAIT_POLICY':'PASSIVE','K200_CTX':'4096','K200_BATCH':'128','K200_UBATCH':'128','K200_INPUT_IDS':str(O/'inputs/prompt2048.ids'),'K200_TEACHER_FILE':str(O/'inputs/teacher64.ids'),'K200_PP_LENGTHS':'128,512','K200_WARMUP':'1','K200_REPEATS':'3','K200_IGNORE_EOS':'1','GGML_KUNLUN_COMMON_Q4':'1','KUNLUN_BROADCAST_VECTOR':'1','GGML_KUNLUN_ROPE':'1','GGML_KUNLUN_SWIGLU':'0','GGML_KUNLUN_ATTENTION':'1','KUNLUN_ATTENTION_MATMUL':'auto','KUNLUN_ATTENTION_SOFTMAX':'fast','KUNLUN_ATTENTION_PREP':'auto','KUNLUN_FLASH_KV':'1','K200_FLASH':'1','GGML_KUNLUN_FLASH':'1'}
from quality import quality
data=[]
for pd in [0,1]:
 for mode in (['reuse1024','legacy'] if pd==0 else ['legacy','reuse1024']):
  name=f'repeat-7b-short-p{pd}-{mode}';text=runner.run(name,[R/'build/flash-kq-llama/bin/kunlun-llm-smoke',R/'models/llama2-7b-q4_0.gguf','none',16,O/(name+'.bin'),pd],{**base,'KUNLUN_FLASH_KQ':mode},1200);print(name,'\n'+'\n'.join(l for l in text.splitlines() if l.startswith('K200PERF')),flush=True)
 for n in [128,512]:
  a=O/f'repeat-7b-short-p{pd}-legacy.bin.p{n}';b=O/f'repeat-7b-short-p{pd}-reuse1024.bin.p{n}';same=a.read_bytes()==b.read_bytes();assert same;data.append(dict(pd=pd,prefix=n,bitwise=same,**quality(a,b)))
 (O/'repeat-7b-quality.json').write_text(json.dumps(data,indent=2))
print('REPEAT 7B SHORT COMPLETE',flush=True)
