from pathlib import Path
import runpy
O=Path('/home/unagidojyou/kunlun-k200/results/flash-decoding-20261026')
for name in ['collect-final.py','make-report.py']:runpy.run_path(str(O/name),run_name='__main__')
