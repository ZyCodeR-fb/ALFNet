"""Read-only config audit; output is generated evidence, no model/data loading."""
import hashlib
import json
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[4]
WORK=ROOT/'outputs_v4/wave_fusion_workspace_20260913'


def main():
    plan=json.loads((WORK/'configs/paper/metrla_wave_screen_v1.json').read_text())
    base=yaml.safe_load((WORK/'configs/paper/metrla_seed32_nine_models/METRLA_ALFNet.yaml').read_text())
    source=yaml.safe_load((WORK/'configs/paper/corrected_baseline_seed32/PEMSBAY_GWNET.yaml').read_text())
    assert plan['datasets']==['METRLA'] and plan['seeds']==[32] and len(plan['runs'])==5
    checked=[]
    for run in plan['runs']:
        v=run['variant'];c=yaml.safe_load((WORK/run['config']).read_text())
        assert c['dataset']==base['dataset']
        t=c['training'];assert (t['batch_size'],t['num_workers'],t['max_epoch'],t['patience'],t['early_stopping_start_epoch'])==(64,0,100,10,40)
        assert t['loss']==('mae' if v.endswith('mae') else 'mse')
        if v=='gwnet_mae':
            assert c['lifecycle']==source['lifecycle'] and c['family']==source['family']
            assert not c['runtime']['t_mark']
            assert t['lr']==source['training']['lr'] and t['teacher_forcing_mode']=='free_run'
        else:
            assert c['lifecycle']==base['lifecycle'] and c['family']['model']==base['family']['model']
            expected=dict(base['family']['recipe'])
            if v=='temporal_mse':expected['graph_temporal_residual']=True
            if v.startswith('wave_'):expected['predictor_mode']='source_wave'
            assert c['family']['recipe']==expected
            assert t['lr']==base['training']['lr']
            assert t['teacher_forcing_mode']==('free_run' if v.startswith('wave_') else 'linear_decay')
            for expert in ('graph','patch'):
                old=yaml.safe_load((WORK/base['family']['experts'][expert]['config_path']).read_text())
                new=yaml.safe_load((WORK/c['family']['experts'][expert]['config_path']).read_text())
                assert old==new
        checked.append(v)
    assert (ROOT/'references/GWNet/model.py').read_bytes()==(WORK/'model/comparison_model/GWNet/source_model.py').read_bytes()
    report=dict(passed=True,plan_sha256=plan['plan_sha256'],variants=checked,
                source_copy_exact=True,test_accessed=False,pemsbay_scheduled=False,
                audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out=ROOT/plan['output_root']/plan['plan_sha256'];out.mkdir(parents=True,exist_ok=True)
    (out/'config_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
