"""Paired, validation-only METR-LA support interventions; no fitting or selection.

Lifecycle trajectories reuse the loader's state construction. Communication-mask
layouts retain a mature active role: packet deletion is not sensor onboarding.
Only the focal node's input columns change. Predictions rerun reconstruction.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.v4_runner import build_v4_session, _forward, to_device
from scripts.v4_utils import sha256_file

Q_VALUES = (0, 3, 6, 9, 12)
LAYOUTS = ('recent', 'early', 'dispersed', 'middle_gap')
TRAJECTORIES = {'full': ('steady', 0), 'join_3': ('join', 3),
                'join_6': ('join', 6), 'fault_3': ('fault', 3),
                'fault_6': ('fault', 6), 'outage': ('fault', 12)}
INPUT_FIELDS = ('base_visible_input', 'input_available_mask',
                'decoder_available_mask', 'lifecycle_role_ids',
                'lifecycle_state_age', 'lifecycle_context')


def layout_mask(q, layout, length=12):
    if q not in range(length + 1) or layout not in LAYOUTS:
        raise ValueError((q, layout))
    mask = np.zeros(length, dtype=bool)
    if q == 0:
        return mask
    if layout == 'recent':
        indices = np.arange(length - q, length)
    elif layout == 'early':
        indices = np.arange(q)
    elif layout == 'dispersed':
        indices = np.floor((np.arange(q) + .5) * length / q).astype(int)
    else:
        # One centered contiguous gap; observations flank it.
        mask[:] = True
        start = q // 2
        mask[start:start + length - q] = False
        return mask
    mask[indices] = True
    return mask


def intervene(dataset, index, node, scenario='steady', evidence=0, keep=None):
    if node not in dataset.context.signal_scope:
        raise ValueError('Focal node outside validation signal scope')
    base = dataset[index]
    entry = dict(dataset.schedule[index], scenario=scenario,
                 evidence_steps=evidence, changed_nodes=[node])
    arm = dataset._sample_from_entry(index, entry)
    result = dict(base)
    for key in INPUT_FIELDS:
        result[key] = base[key].clone()
        result[key][:, node] = arm[key][:, node]
    if keep is not None:
        keep = torch.as_tensor(keep, dtype=torch.bool)
        # Intersection, never synthesis of a naturally absent measurement.
        available = result['input_available_mask'][:, node] & keep
        result['input_available_mask'][:, node] = available
        result['base_visible_input'][:, node, 0] = torch.where(
            available, result['base_visible_input'][:, node, 0], 0.)
        result['lifecycle_context'][:, node, -1] = available.float()
        result['decoder_available_mask'][:, node] = (
            available[-1] & result['target_available_mask'][:, node])
    others = torch.arange(dataset.num_nodes) != node
    for key in INPUT_FIELDS:
        assert torch.equal(result[key][:, others], base[key][:, others]), key
    for key in ('target', 'target_available_mask', 'target_cohort_mask',
                'loss_mask', 'encoder_time_marks', 'decoder_time_marks'):
        assert torch.equal(result[key], base[key]), key
    start = int(base['origin'])
    natural = torch.from_numpy(dataset.natural[start:start+12, node])
    assert not (result['input_available_mask'][:, node] & ~natural).any()
    assert torch.isfinite(result['base_visible_input']).all()
    return result


def select_pairs(dataset, per_track, seed=20260908):
    rng = np.random.default_rng(seed)
    pairs = []
    # Uniform schedule candidates within the two pre-existing event tracks;
    # choose a focal cohort node without inspecting any prediction/error.
    for track in ('join', 'fault'):
        candidates = [i for i,e in enumerate(dataset.schedule) if e['scenario'] == track]
        count = 0
        for i in rng.permutation(candidates):
            start = int(dataset.origins[i])
            nodes = [int(n) for n in dataset.schedule[i]['changed_nodes']
                     if dataset.natural[start:start+24, n].all()]
            if not nodes:
                continue
            node = int(rng.choice(nodes))
            assert node in dataset.context.signal_scope
            pairs.append(dict(pair_id=len(pairs), index=int(i), node=node,
                              origin=start, track=track,
                              regime=dataset.schedule[i]['calendar_regime']))
            count += 1
            if count == per_track:
                break
        if count != per_track:
            raise ValueError('Insufficient naturally complete eligible windows')
    return pairs


def arms():
    for label,(scenario,evidence) in TRAJECTORIES.items():
        yield dict(experiment='trajectory', condition=label, q=-1, layout='',
                   scenario=scenario, evidence=evidence, keep=None)
    for q in Q_VALUES:
        # q=0 and q=12 are single anchors, not four independent replications.
        for layout in (('recent',) if q in (0,12) else LAYOUTS):
            yield dict(experiment='support', condition=f'q{q}_{layout}', q=q,
                       layout='all' if q in (0,12) else layout,
                       scenario='steady', evidence=0, keep=layout_mask(q,layout))


def write_csv(path, rows):
    with Path(path).open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summarize(rows):
    results=[]
    for model in sorted({r['model'] for r in rows}):
        for condition in dict.fromkeys(r['condition'] for r in rows):
            for track in ('all','join','fault'):
                group=[r for r in rows if r['model']==model and r['condition']==condition
                       and (track=='all' or r['track']==track)]
                if not group: continue
                r=group[0]
                results.append(dict(model=model,experiment=r['experiment'],condition=condition,
                    q=r['q'],layout=r['layout'],track=track,pairs=len(group),
                    MAE=float(np.mean([x['MAE'] for x in group])),
                    RMSE=float(np.sqrt(np.mean([x['MSE'] for x in group]))),
                    MAPE=float(np.mean([x['MAPE'] for x in group]))))
    return results


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(32); np.random.seed(32)
    out=Path(args.output)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    records=json.loads((ROOT/'outputs_v4/reports/support_analysis_v1/case_records.json').read_text())
    cp=next(r for r in records if r['dataset']=='METRLA')['checkpoints']
    snapshot=json.loads((ROOT/'paper/experiments_evidence_snapshot.json').read_text())
    mask_run=next(r for r in snapshot['runs'] if r['dataset']=='METRLA' and r['variant']=='mask_only')
    specs={
        'ALFNet':('configs/paper/metrla_seed32_nine_models/METRLA_ALFNet.yaml',cp['ALFNet']['path']),
        'MoGERNN':('configs/paper/metrla_seed32_nine_models/METRLA_MoGERNN.yaml',cp['MoGERNN']['path']),
        'Mask-only':(mask_run['config'],str(Path(mask_run['report']).parent/'best.pt'))}
    all_rows=[]; sources={}; pairs=None; reference=None
    arm_specs=list(arms())
    for name,(config,checkpoint_path) in specs.items():
        session=build_v4_session(config,device=args.device,batch_size=args.batch_size,num_workers=0)
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        assert checkpoint['config_sha256']==sha256_file(ROOT/config)
        session.model.load_state_dict(checkpoint['model_state'],strict=True)
        session.model.to(args.device).eval()
        dataset=session.validation_loader.dataset
        assert session.validation_context.phase=='validation'
        assert checkpoint['scaler']==dataset.scaler
        if pairs is None:
            pairs=select_pairs(dataset,args.per_track)
            reference=[dataset[p['index']] for p in pairs]
            (out/'pairs.json').write_text(json.dumps(pairs,indent=2))
            manifest=dict(split='validation',dataset='METRLA',selection_seed=20260908,
                per_track=args.per_track,selection='random eligible schedule origin, random focal cohort node; natural history/future complete',
                eligibility='12 naturally available history and 12 future targets; no fabricated readings',
                support_state='AAS age=1 throughout; layouts are communication deletion, not lifecycle relabeling',
                trajectory_state='loader canonical steady/join/fault roles and age',
                context='all nonfocal inputs, graph, targets, calendar, scope unchanged per pair',
                aggregation='equal paired windows and 12 horizons; pooled RMSE; not LifeAvg',
                q_values=Q_VALUES,layouts=LAYOUTS,trajectories=TRAJECTORIES,
                device=args.device,script_sha256=sha256_file(Path(__file__)))
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        for p,base in zip(pairs,reference):
            other=dataset[p['index']]
            for key in INPUT_FIELDS+('target','target_available_mask'):
                assert torch.equal(base[key],other[key]), (name,key)
        sources[name]=dict(config=config,config_sha256=sha256_file(ROOT/config),
            checkpoint=checkpoint_path,checkpoint_sha256=sha256_file(checkpoint_path),epoch=checkpoint['epoch'])
        rows=[]
        for arm in arm_specs:
            for start in range(0,len(pairs),args.batch_size):
                part=pairs[start:start+args.batch_size]
                items=[intervene(dataset,p['index'],p['node'],arm['scenario'],arm['evidence'],arm['keep']) for p in part]
                with torch.inference_mode():
                    output=_forward(session,to_device(default_collate(items),args.device),
                                    session.validation_context,checkpoint['epoch'],None)
                pred=output['pred'].detach().cpu().numpy()
                for j,(p,item) in enumerate(zip(part,items)):
                    n=p['node']; std=dataset.scaler['std']; mean=dataset.scaler['mean']
                    actual=item['target'][:,n,0].numpy()*std+mean
                    forecast=pred[j,:,n,0]*std+mean; difference=forecast-actual
                    q=int(item['input_available_mask'][:,n].sum())
                    if arm['q']>=0: assert q==arm['q']
                    rows.append(dict(model=name,experiment=arm['experiment'],condition=arm['condition'],
                        layout=arm['layout'],q=q,**p,
                        mask=''.join(str(int(x)) for x in item['input_available_mask'][:,n]),
                        MAE=float(np.abs(difference).mean()),MSE=float(np.square(difference).mean()),
                        RMSE=float(np.sqrt(np.square(difference).mean())),
                        MAPE=float((np.abs(difference)/np.maximum(np.abs(actual),1e-6)).mean()*100),
                        prediction=json.dumps(forecast.tolist()),target=json.dumps(actual.tolist())))
            print(name,arm['condition'],'pairs',len(pairs),flush=True)
        write_csv(out/(name.replace('-','_')+'_paired.csv'),rows)
        all_rows.extend(rows)
        del session
    write_csv(out/'paired_predictions.csv',all_rows)
    summary=summarize(all_rows);write_csv(out/'metrics.csv',summary)
    lookup={(r['model'],r['condition'],r['pair_id']):r for r in all_rows}
    contrasts=[]
    for r in all_rows:
        b=lookup['MoGERNN',r['condition'],r['pair_id']]
        anchor='full' if r['experiment']=='trajectory' else 'q12_recent'
        full=lookup[r['model'],anchor,r['pair_id']]
        contrasts.append({k:r[k] for k in ('model','experiment','condition','q','layout','pair_id','track')} |
            dict(MAE=r['MAE'],baseline_MAE=b['MAE'],paired_MAE_gain=b['MAE']-r['MAE'],
                 MAE_change_from_full=r['MAE']-full['MAE']))
    write_csv(out/'paired_contrasts.csv',contrasts)
    (out/'sources.json').write_text(json.dumps(sources,indent=2))
    (out/'COMPLETE').write_text('Completed validation-only paired inference; no training.\n')
    print('COMPLETE',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs_v4/reports/observation_support_v1')
    parser.add_argument('--per-track',type=int,default=64)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--device',default='cpu')
    run(parser.parse_args())
