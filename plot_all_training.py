import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def get_scalars(event_file, tag):
    """Extract scalar values for a given tag"""
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    try:
        scalars = ea.Scalars(tag)
        steps = [s.step for s in scalars]
        values = [s.value for s in scalars]
        return steps, values
    except:
        return [], []

def plot_training_curves(exp_name, fold, output_dir):
    """Plot training curves for a single fold"""
    base_dir = '/home/shanyiye/CLAM/results'
    fold_dir = os.path.join(base_dir, exp_name, str(fold))
    
    if not os.path.exists(fold_dir):
        print(f"  Fold {fold} directory not found, skipping...")
        return False
    
    event_files = [f for f in os.listdir(fold_dir) if f.startswith('events')]
    if not event_files:
        print(f"  No event file found in fold {fold}, skipping...")
        return False
    
    event_file = os.path.join(fold_dir, event_files[0])
    
    tags = ['train/loss', 'train/class_0_acc', 'val/loss', 'val/auc', 'val/error']
    tag_names = ['Train Loss', 'Train Acc (Class 0)', 'Val Loss', 'Val AUC', 'Val Error']
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'{exp_name} - Fold {fold}', fontsize=14, fontweight='bold')
    
    data_found = False
    for i, (tag, name) in enumerate(zip(tags, tag_names)):
        ax = axes[i // 2, i % 2]
        steps, values = get_scalars(event_file, tag)
        
        if steps:
            data_found = True
            ax.plot(steps, values, marker='o', markersize=3, linewidth=1.5)
            ax.set_xlabel('Step', fontsize=10)
            ax.set_ylabel(name, fontsize=10)
            ax.set_title(name, fontsize=11)
            ax.grid(True, alpha=0.3)
            
            min_val = min(values)
            min_idx = values.index(min_val)
            if 'loss' in tag or 'error' in tag:
                ax.axhline(y=min_val, color='r', linestyle='--', alpha=0.5, label=f'Min: {min_val:.4f}')
            else:
                ax.axhline(y=max(values), color='g', linestyle='--', alpha=0.5, label=f'Max: {max(values):.4f}')
            ax.legend(fontsize=8)
        else:
            ax.set_title(f'{name} (No data)', fontsize=11)
    
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'fold_{fold}.png')
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    return data_found

def main():
    base_dir = '/home/shanyiye/CLAM/results'
    output_base = '/home/shanyiye/CLAM/analysis_results/training_curves'
    
    experiments = [
        'dlbcl_gcb_nongcb_clam_mb_nows_s1',
        'dlbcl_gcb_nongcb_clam_mb_s1',
        'dlbcl_gcb_nongcb_clam_sb_nows_s1',
        'dlbcl_gcb_nongcb_clam_sb_s1',
        'dlbcl_gcb_nongcb_mil_s1',
    ]
    
    for exp_name in experiments:
        output_dir = os.path.join(output_base, exp_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nProcessing: {exp_name}")
        
        success_count = 0
        for fold in range(10):
            print(f"  Processing fold {fold}...", end=" ")
            if plot_training_curves(exp_name, fold, output_dir):
                success_count += 1
                print("OK")
            else:
                print("Skipped")
        
        print(f"  Completed {success_count}/10 folds")

    print(f"\n\nAll training curves saved to: {output_base}")

if __name__ == '__main__':
    main()
