from pathlib import Path
import os,sys,json,hashlib
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-online-20261026';A=R/'results/flash-attention-20261026';sys.path.insert(0,str(O));import runner;runner.OUT=O
os.environ['LD_LIBRARY_PATH']=':'.join(str(R/x) for x in ['external/xre/lib64','external/xdnn/so','external/xtdk/shlib'])
base={'OMP_WAIT_POLICY':'PASSIVE','GGML_KUNLUN_COMMON_Q4':'1','KUNLUN_BROADCAST_VECTOR':'1','GGML_KUNLUN_ROPE':'1','GGML_KUNLUN_SWIGLU':'0','GGML_KUNLUN_ATTENTION':'1','KUNLUN_ATTENTION_MATMUL':'auto','KUNLUN_ATTENTION_SOFTMAX':'fast','KUNLUN_ATTENTION_PREP':'auto','KUNLUN_FLASH_KV':'1','K200_QUALITY_BATCH':'128','K200_FLASH':'1','GGML_KUNLUN_FLASH':'1','KUNLUN_FLASH_ONLINE':'ordered'}
results=[]
for model,file in [('tiny','tinyllama-q4_0.gguf'),('7b','llama2-7b-q4_0.gguf')]:
 name=f'corpus-{model}-ordered';ref=A/f'corpus-final-{model}.bin';out=O/(name+'.bin')
 t=runner.run(name,[R/'build/flash-online-llama/bin/kunlun-quality-eval',R/'models'/file,'none',O/'inputs/corpus1024.ids',256,out,ref,A/f'corpus-{model}-cpuattn.bin'],base,1200)
 same=out.read_bytes()==ref.read_bytes();row=dict(test=name,bitwise=same,lines=[l for l in t.splitlines() if l.startswith(('QUALITY ','CORPUS '))],sha256=hashlib.sha256(out.read_bytes()).hexdigest());results.append(row);print(row,flush=True)
 (O/'ordered-corpus-quality.json').write_text(json.dumps(results,indent=2));assert same
print('ORDERED CORPUS COMPLETE',flush=True)
