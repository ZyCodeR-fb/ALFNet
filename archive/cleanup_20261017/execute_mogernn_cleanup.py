from pathlib import Path
import json,tarfile,hashlib
ROOT=Path('/media/xd/38162e9d-06bd-4953-8790-0bc340486a80/ZMY/MoGERNN')
out=ROOT/'archive/cleanup_20261017';p=json.loads((out/'plan.json').read_text())
members=set(json.loads((out/'archive_members.json').read_text()))
remove=set(p['remove']);excluded={'Merlin','STFiT','STGODE','GWNET','HA','FCLSTM','MCPST_FSL','ALFWave'}
for row in p['weights']:
 if row.get('model') in excluded or excluded.intersection(Path(row['path']).parts):
  row['action']='delete_excluded_model_weight';remove.add(row['path'])
# Archive and remove metadata only when the complete run has no retained weight.
retain={row['path'] for row in p['weights'] if row['action']=='retain'}
for row in p['weights']:
 if row['action']=='retain':continue
 folder=ROOT/row['path'];folder=folder.parent
 if any(str(x.relative_to(ROOT)) in retain for x in folder.rglob('*.pt')):continue
 for x in folder.rglob('*'):
  if x.is_file() and str(x.relative_to(ROOT)) in members:remove.add(str(x.relative_to(ROOT)))
for d in ['FC-LSTM','GWNET','HA','MCPST-FSL','ST-FiT','STGODE']:
 for x in (ROOT/'configs'/d).rglob('*'):
  if x.is_file() and str(x.relative_to(ROOT)) in members:remove.add(str(x.relative_to(ROOT)))
# Guard all explicitly frozen final checkpoint references.
manifest=json.loads((ROOT/'paper/pems08_wd_test_20261012/selected_configuration_manifest.json').read_text())
for row in manifest['entries'].values():
 assert row['checkpoint'] not in remove
 q=ROOT/row['checkpoint'];assert hashlib.file_digest(q.open('rb'),'sha256').hexdigest()==row['checkpoint_sha256']
deleted=[]
lookup={r['path']:r for r in p['weights']}
for rel in sorted(remove):
 q=ROOT/rel
 if not q.is_file():continue
 if q.suffix=='.pt':
  row=lookup[rel];assert row['action']!='retain'
  assert hashlib.file_digest(q.open('rb'),'sha256').hexdigest()==row['sha256']
 else:assert rel in members or '__pycache__' in rel,rel
 deleted.append(dict(path=rel,bytes=q.stat().st_size));q.unlink()
for d in ['model','configs','artifacts','scripts','tests']:
 for q in sorted((ROOT/d).rglob('*'),key=lambda x:len(x.parts),reverse=True):
  if q.is_dir() and not any(q.iterdir()):q.rmdir()
p['remove']=sorted(remove);p['deleted']=deleted
(out/'plan.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
summary={'deleted_files':len(deleted),'deleted_bytes':sum(x['bytes'] for x in deleted),'deleted_weights':sum(r['action']!='retain' for r in p['weights']),'retained_weights':len(retain)}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(summary)
