from pathlib import Path
import json, yaml, hashlib, ast

ROOT=Path('/media/xd/38162e9d-06bd-4953-8790-0bc340486a80/ZMY/MoGERNN')
OUT=ROOT/'archive/cleanup_20261017'
OUT.mkdir(parents=True,exist_ok=True)
obsolete={'Merlin','STFiT','STGODE','GWNET','HA','FCLSTM','MCPST_FSL','ALFWave'}
retired_dirs=['model/comparison_model/'+x for x in ['Merlin','STFiT','stgode','GWNet','HA','FCLSTM','MCPST_FSL','GCN-M']]
retired_dirs+=['model/ALFWave']
retired_tests={'tests/test_merlin_adapter.py','tests/test_merlin_stages.py'}
remove=set()
for d in retired_dirs:
 remove.update(str(p.relative_to(ROOT)) for p in (ROOT/d).rglob('*') if p.is_file())
remove.update(retired_tests)
remove.add('scripts/train_merlin_v4.py')
weights=[]
for p in sorted((ROOT/'artifacts').rglob('*.pt')):
 if not p.is_file():continue
 rel=str(p.relative_to(ROOT));cfg=None;cfgpath=None
 for q in [p.parent/'config.original.yaml',p.parent/'config.yaml']:
  if q.exists():cfg=yaml.safe_load(q.read_text());cfgpath=str(q.relative_to(ROOT));break
 report=p.parent/'run_report.json'
 if cfg is None and report.exists():
  q=ROOT/json.loads(report.read_text()).get('config','__missing__')
  if q.is_file():cfg=yaml.safe_load(q.read_text());cfgpath=str(q.relative_to(ROOT))
 cfg=cfg or {};runtime=cfg.get('runtime',{});variant=runtime.get('model_variant','');loss=cfg.get('training',{}).get('loss')
 # All final-family experiments and every ablation are protected, including
 # rejected objectives/HP trials. Only conclusively pre-final history is removed.
 old=rel.startswith('artifacts/models/history/') and runtime.get('model')=='ALFNet' and loss!='mae_focal_raw02' and 'ablation' not in variant.lower()
 # History-linear heads are pre-final architecture even with the focal objective.
 old=old or (rel.startswith('artifacts/models/history/') and runtime.get('model')=='ALFNet' and any(s in variant for s in ['linear32','linear42','history_completion']))
 if old:remove.add(rel)
 weights.append(dict(path=rel,bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),action='delete_pre_final_weight' if old else 'retain',config=cfgpath,model=runtime.get('model'),variant=variant,loss=loss))
# Retain all scripts required by current figures, reproducibility tests, and all
# final-family studies. Retire only unreferenced one-off reports/old plotters.
scripts={p.stem:p for p in (ROOT/'scripts').glob('*.py')}
roots=[p for p in scripts.values() if not p.name.startswith(('analyze_','plot_','diagnose_','summarize_','repair_','resume_'))]
roots += [ROOT/'scripts'/n for n in ['plot_final_figures_20261017.py','plot_figure_hparams_20261017.py','plot_advisor_experiments_20261012.py']]
roots += [p for p in (ROOT/'tests').glob('*.py') if str(p.relative_to(ROOT)) not in retired_tests]
seen=set();stack=roots[:]
while stack:
 p=stack.pop()
 if p in seen or not p.exists():continue
 seen.add(p)
 text=p.read_text()
 for name,q in scripts.items():
  if q!=p and (('scripts.'+name) in text or ('import '+name) in text or (name+'.py') in text):stack.append(q)
for p in scripts.values():
 if p not in seen:remove.add(str(p.relative_to(ROOT)))
plan=dict(policy='All final-family weights and all ablations retained; only verified pre-final ALFNet history weights deleted. Removed sources/evidence archived.',remove=sorted(x for x in remove if (ROOT/x).is_file()),weights=weights)
(OUT/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'remove_files':len(plan['remove']),'delete_weights':sum(x['action']!='retain' for x in weights),'retain_weights':sum(x['action']=='retain' for x in weights),'delete_weight_bytes':sum(x['bytes'] for x in weights if x['action']!='retain'),'retire_scripts':[x for x in plan['remove'] if x.startswith('scripts/')]}))
