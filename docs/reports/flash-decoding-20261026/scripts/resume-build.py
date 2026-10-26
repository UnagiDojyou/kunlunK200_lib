from pathlib import Path
import os,json,subprocess,hashlib
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026'
env=os.environ.copy();env.update(XRE=str(R/'external/xre'),XDNN=str(R/'external/xdnn'),XTDK_DIR=str(R/'external/xtdk'))
subprocess.run(['cmake','--build',str(R/'build/flash-decode-llama'),'-j4','--target','kunlun-flash-decoding-bench','kunlun-llm-smoke','kunlun-quality-eval','kunlun-attention-test','kunlun-swiglu-test','kunlun-backend-contract'],env=env,check=True)
D=R/'build/flash-decode-llama/kunlunk200/lib/kunlunk200/kernels';after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('*.bin')};assert after==json.loads((O/'pre-adoption-images.json').read_text())
print('ADOPTED; measured shader images unchanged; existing custom llama tools built')
