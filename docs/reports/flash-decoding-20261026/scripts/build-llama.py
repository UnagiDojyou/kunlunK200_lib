from pathlib import Path
import os,subprocess
R=Path('/home/unagidojyou/kunlun-k200');L=R/'repos/KunlunK200'
p=L/'src/flash-decode.inc';s=p.read_text().replace('if(chunks>tiles)chunks=tiles;if(chunks<1','if(chunks>tiles)chunks=tiles;\n if(chunks<1').replace('++step){int unused=0;','++step){');p.write_text(s)
env=os.environ.copy();env.update(XRE=str(R/'external/xre'),XDNN=str(R/'external/xdnn'),XTDK_DIR=str(R/'external/xtdk'),LD_LIBRARY_PATH=':'.join(str(R/x) for x in ['external/xre/lib64','external/xdnn/so','external/xtdk/shlib']))
subprocess.run(['cmake','-S',str(R/'repos/llama.cpp'),'-B',str(R/'build/flash-decode-llama'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DGGML_KUNLUN=ON','-DLLAMA_BUILD_EXAMPLES=ON','-DLLAMA_BUILD_TESTS=ON','-DLLAMA_BUILD_SERVER=OFF','-DLLAMA_BUILD_APP=OFF','-DLLAMA_BUILD_MTMD=OFF','-DKUNLUN_USE_SYSTEM_LIBRARY=OFF'],env=env,check=True)
subprocess.run(['cmake','--build',str(R/'build/flash-decode-llama'),'-j6','--target','kunlun-llm-smoke','kunlun-quality-eval','kunlun-attention-test','kunlun-swiglu-test','kunlun-backend-contract'],env=env,check=True)
subprocess.run(['cmake','--build',str(R/'build/flash-decode-lib'),'-j6','--target','k200-flash-decode-bench','k200-flash-contract','k200-attention-contract','k200-common-q4-layout'],env=env,check=True)
