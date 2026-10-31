import json, math, shutil, subprocess, pandas as pd
R = '/Users/apple/Desktop/squares_in_squares'
L = {int(k): v for k, v in json.load(open(f'{R}/watch/iso_fixed2_L.json')).items()}
mine = sum(L[n] / math.sqrt(n) - 1 for n in L); cur = float(open(f'{R}/watch/best_score.txt').read())
print(f'mine {mine:.10f} | current best_score.txt {cur:.10f}')
if mine < cur - 1e-9 and not any(v <= 0 for v in L.values()):
    shutil.copy(f'{R}/watch/fixed2_candidate.csv', f'{R}/watch/best_current.csv'); json.dump(L, open(f'{R}/watch/best_L.json', 'w')); open(f'{R}/watch/best_score.txt', 'w').write(repr(mine))
    shutil.copy(f'{R}/watch/fixed2_candidate.csv', f'{R}/watch/reports/best_20261031_isolated_{mine:.8f}.csv')
    print('PROMOTED'); print(subprocess.run([f'{R}/.venv/bin/python', f'{R}/update_summary_refresh.py'], capture_output=True, text=True).stdout[-400:])
else: print('not promoted (current is at least as good)')
