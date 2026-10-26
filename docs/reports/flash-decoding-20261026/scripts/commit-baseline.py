from pathlib import Path
import subprocess,json
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026'
records={}
for name,repo,message in [('library',R/'repos/KunlunK200','kunlun: optimize FlashAttention online softmax and KQ reuse (0.7.2)'),('llama',R/'repos/llama.cpp','kunlun: record FlashAttention 0.7.2 library and evaluation guides')]:
 def git(*a):return subprocess.check_output(['git',*a],cwd=repo,text=True)
 subprocess.run(['git','diff','--check'],cwd=repo,check=True)
 subprocess.run(['git','add','-A'],cwd=repo,check=True)
 print(name,git('diff','--cached','--stat'),flush=True)
 subprocess.run(['git','commit','-m',message,'-m','Assisted-by: OpenAI Codex'],cwd=repo,check=True)
 records[name]={'head':git('rev-parse','HEAD').strip(),'branch':git('branch','--show-current').strip(),'status':git('status','--short')};assert not records[name]['status']
 subprocess.run(['git','switch','-c','codex/k200-flash-decoding-20261026'],cwd=repo,check=True)
(O/'baseline-commits.json').write_text(json.dumps(records,indent=2));print('BASELINE',json.dumps(records))
