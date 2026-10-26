from pathlib import Path
import json
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026'
for p in (R/'repos/llama.cpp').rglob('*kunlun*.cpp'):
 if 'vendor' not in p.parts and any(s in p.name for s in ['smoke','quality']):
  print(p);(O/p.name).write_text(p.read_text())
