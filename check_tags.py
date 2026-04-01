import os
from tensorboard.backend.event_processing import event_accumulator

def check_available_tags(exp_name, fold='0'):
    base_dir = '/home/shanyiye/CLAM/results'
    fold_dir = os.path.join(base_dir, exp_name, fold)
    
    event_files = [f for f in os.listdir(fold_dir) if f.startswith('events')]
    if not event_files:
        return None
    
    event_file = os.path.join(fold_dir, event_files[0])
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    return ea.Tags()

experiments = [
    'dlbcl_gcb_nongcb_clam_mb_nows_s1',
    'dlbcl_gcb_nongcb_clam_mb_s1',
    'dlbcl_gcb_nongcb_clam_sb_nows_s1',
    'dlbcl_gcb_nongcb_clam_sb_s1',
    'dlbcl_gcb_nongcb_mil_s1',
]

for exp in experiments:
    print(f"\n=== {exp} ===")
    tags = check_available_tags(exp)
    if tags:
        print("Scalars:", tags.get('scalars', []))
