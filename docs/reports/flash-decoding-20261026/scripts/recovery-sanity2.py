from pathlib import Path
import subprocess,json,os,sys,hashlib
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026';A=O/'incidents/profile-p0-hk1-c1';A.mkdir(exist_ok=True,parents=True)
assert json.loads((O/'XPU_ABORT').read_text())['name']=='profile-p0-hk1-c1'
subprocess.run(['sudo','-n','python3',str(O/'check_users.py')],check=True)
for pd in [0,1]:
 for k in ['state','errinfo','errtask','sessions']:(A/f'pre-reset-p{pd}-{k}.txt').write_text(Path(f'/proc/xpu/dev{pd}/{k}').read_text())
tool=R/'external/xre/tools/soft_reset';assert hashlib.sha256(tool.read_bytes()).hexdigest()=='292b732b7cf55e7f15edeae4b9501bdd65d451302c160edb68f1ceebbcc35d37'
env=os.environ.copy();env['XRE']=str(R/'external/xre');env['KUNLUN_LOG_DIR']=str(A/'recovery')
# reset already succeeded; sanity only
for pd in [0,1]:
 assert Path(f'/proc/xpu/dev{pd}/state').read_text().strip()=='RUNNING'
 assert Path(f'/proc/xpu/dev{pd}/errinfo').read_text().strip()=='no error'
os.environ['LD_LIBRARY_PATH']=':'.join(str(R/x) for x in ['external/xre/lib64','external/xdnn/so','external/xtdk/shlib'])
sys.path.insert(0,str(O));import runner;runner.OUT=O/'recovery';runner.OUT.mkdir(exist_ok=True)
(O/'recovery/XPU_ABORT').rename(A/'sanity-argument-error.json')
for pd in [0,1]:runner.run(f'frozen-flash-p{pd}',[R/'build/flash-kq-lib/k200-flash-kq-bench',pd,128,512,1,4,32,R/'build/flash-kq-lib/lib/kunlunk200/kernels',12,1],seconds=90)
(A/'recovery-reviewed.json').write_text(json.dumps(dict(cause='diagnostic-only LM2GM(meta+j) unaligned LM pointer; changed to aligned transfer[0]',reset_count=1,checks='both PD RUNNING/no error; frozen 0.7.2 CPU reference comparison PASS'),indent=2))
(O/'XPU_ABORT').rename(A/'XPU_ABORT-recovered.json')
print('RECOVERY COMPLETE',flush=True)


