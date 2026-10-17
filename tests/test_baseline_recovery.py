from scripts.recover_baseline_extension import isolate_failed_families

def test_failure_does_not_block_unrelated_models_or_interrupt_active_jobs():
    configs={k:{'model':m} for k,m in [('03_GinAR','GinAR'),('04_GinAR','GinAR'),
            ('08_GinAR','GinAR'),('03_STID','STID'),('03_Merlin','Merlin')]}
    jobs=[{'key':k,'status':s} for k,s in [('03_GinAR','failed'),('04_GinAR','queued'),
            ('08_GinAR','running'),('03_STID','queued'),('03_Merlin','complete')]]
    isolate_failed_families(jobs,configs)
    assert [j['status'] for j in jobs]==['failed','blocked_same_model','running','queued','complete']
    isolate_failed_families(jobs,configs)
    assert jobs[0]['status']=='failed' and jobs[3]['status']=='queued'
