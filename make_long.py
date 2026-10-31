import json, glob, sys, os, shutil
S, LIM, C, tag = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
src = 'pub3/lightning'; dst = 'kernel_long'; shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
nb = json.load(open(glob.glob(f'{src}/*.ipynb')[0])); hits = 0
for c in nb['cells']:
    if c['cell_type'] != 'code': continue
    s = ''.join(c['source']); o = s
    s = s.replace("RUN_SEARCH=False", "RUN_SEARCH=True")
    s = s.replace("shutil.copyfile(DATA/'projected_public.csv','artifacts/update25_public_submission.csv')",
                  "OURS=next(Path('/kaggle/input').rglob('final_enforced.csv'));shutil.copyfile(OURS,'artifacts/update25_public_submission.csv');shutil.copyfile(OURS,'submission.csv');print('baseline = ours:',OURS)")
    s = s.replace("'--seconds','400','--limit','80'],check=True,timeout=430", f"'--seconds','{S}','--limit','{LIM}'],check=True,timeout={int(S)+120}")
    s = s.replace("'--seconds','160','--bins','24','--tag','optional_composed'],check=True,timeout=190", f"'--seconds','{C}','--bins','24','--tag','optional_composed'],check=True,timeout={int(C)+120}")
    if s != o: hits += 1
    c['source'] = s.splitlines(True)
print('cells changed:', hits)
json.dump(nb, open(f'{dst}/squares-cuda-long-search.ipynb', 'w'))
json.dump({"id": "samanyu1808/squares-cuda-long-search", "title": "Squares CUDA long search", "code_file": "squares-cuda-long-search.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
           "dataset_sources": ["lightningv08/squares-cuda-separator-experiment", "cdeotte/squares-in-squares-22-315-data", "lightningv08/squares-in-squares-layout-atlas", "samanyu1808/squares-our-baseline"],
           "competition_sources": ["squares-in-squares"], "kernel_sources": [], "machine_shape": "NvidiaTeslaT4"}, open(f'{dst}/kernel-metadata.json', 'w'), indent=1)
print('built', tag, 'search', S, 's limit', LIM, 'compose', C, 's')
