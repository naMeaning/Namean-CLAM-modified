import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def get_scalars(event_file, tag):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    try:
        scalars = ea.Scalars(tag)
        steps = [s.step for s in scalars]
        values = [s.value for s in scalars]
        return steps, values
    except:
        return [], []

def plot_all_curves(exp_name, fold, output_dir):
    base_dir = '/home/shanyiye/CLAM/results'
    fold_dir = os.path.join(base_dir, exp_name, str(fold))
    
    if not os.path.exists(fold_dir):
        return False
    
    event_files = [f for f in os.listdir(fold_dir) if f.startswith('events')]
    if not event_files:
        return False
    
    event_file = os.path.join(fold_dir, event_files[0])
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    available_tags = ea.Tags().get('scalars', [])
    
    if not available_tags:
        return False
    
    train_tags = [t for t in available_tags if t.startswith('train/')]
    val_tags = [t for t in available_tags if t.startswith('val/') and not t.startswith('val/auc')]
    
    train_loss_tags = [t for t in train_tags if 'loss' in t]
    train_acc_tags = [t for t in train_tags if 'acc' in t]
    train_error_tags = [t for t in train_tags if 'error' in t and 'loss' not in t]
    
    val_loss_tags = [t for t in val_tags if 'loss' in t]
    val_acc_tags = [t for t in val_tags if 'acc' in t]
    val_error_tags = [t for t in val_tags if 'error' in t and 'loss' not in t]
    val_auc_tags = [t for t in available_tags if t == 'val/auc']
    
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle(f'{exp_name} - Fold {fold}\nAll Training Curves', fontsize=16, fontweight='bold')
    
    plot_idx = 1
    
    if train_loss_tags:
        n_loss = len(train_loss_tags)
        fig_temp, axes_loss = plt.subplots(1, n_loss, figsize=(6*n_loss, 5))
        if n_loss == 1:
            axes_loss = [axes_loss]
        for ax, tag in zip(axes_loss, train_loss_tags):
            steps, values = get_scalars(event_file, tag)
            if steps:
                ax.plot(steps, values, marker='o', markersize=3, linewidth=1.5)
                ax.set_xlabel('Step')
                ax.set_ylabel(tag.replace('train/', ''))
                ax.set_title(tag)
                ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'fold_{fold}_train_loss.png'), dpi=120, bbox_inches='tight')
        plt.close()
    
    if train_acc_tags:
        n_acc = len(train_acc_tags)
        fig_temp, axes_acc = plt.subplots(1, n_acc, figsize=(6*n_acc, 5))
        if n_acc == 1:
            axes_acc = [axes_acc]
        for ax, tag in zip(axes_acc, train_acc_tags):
            steps, values = get_scalars(event_file, tag)
            if steps:
                ax.plot(steps, values, marker='o', markersize=3, linewidth=1.5)
                ax.set_xlabel('Step')
                ax.set_ylabel(tag.replace('train/', ''))
                ax.set_title(tag)
                ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'fold_{fold}_train_acc.png'), dpi=120, bbox_inches='tight')
        plt.close()
    
    if train_error_tags:
        n_err = len(train_error_tags)
        fig_temp, axes_err = plt.subplots(1, n_err, figsize=(6*n_err, 5))
        if n_err == 1:
            axes_err = [axes_err]
        for ax, tag in zip(axes_err, train_error_tags):
            steps, values = get_scalars(event_file, tag)
            if steps:
                ax.plot(steps, values, marker='o', markersize=3, linewidth=1.5)
                ax.set_xlabel('Step')
                ax.set_ylabel(tag.replace('train/', ''))
                ax.set_title(tag)
                ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'fold_{fold}_train_error.png'), dpi=120, bbox_inches='tight')
        plt.close()
    
    if val_loss_tags or val_auc_tags:
        n_val = len(val_loss_tags) + len(val_auc_tags)
        fig_temp, axes_val = plt.subplots(1, n_val, figsize=(6*n_val, 5))
        if n_val == 1:
            axes_val = [axes_val]
        
        val_all_tags = val_loss_tags + val_auc_tags
        for ax, tag in zip(axes_val, val_all_tags):
            steps, values = get_scalars(event_file, tag)
            if steps:
                ax.plot(steps, values, marker='o', markersize=3, linewidth=1.5, color='orange')
                ax.set_xlabel('Step')
                ax.set_ylabel(tag.replace('val/', ''))
                ax.set_title(tag)
                ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'fold_{fold}_val_loss_auc.png'), dpi=120, bbox_inches='tight')
        plt.close()
    
    if val_acc_tags or val_error_tags:
        n_comb = len(val_acc_tags) + len(val_error_tags)
        fig_temp, axes_comb = plt.subplots(1, n_comb, figsize=(6*n_comb, 5))
        if n_comb == 1:
            axes_comb = [axes_comb]
        
        val_comb_tags = val_acc_tags + val_error_tags
        for ax, tag in zip(axes_comb, val_comb_tags):
            steps, values = get_scalars(event_file, tag)
            if steps:
                ax.plot(steps, values, marker='o', markersize=3, linewidth=1.5, color='green')
                ax.set_xlabel('Step')
                ax.set_ylabel(tag.replace('val/', ''))
                ax.set_title(tag)
                ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'fold_{fold}_val_acc_error.png'), dpi=120, bbox_inches='tight')
        plt.close()
    
    final_tags = [t for t in available_tags if t.startswith('final/')]
    if final_tags:
        fig_temp, axes_final = plt.subplots(2, (len(final_tags)+1)//2, figsize=(14, 8))
        axes_final = axes_final.flatten()
        for ax, tag in zip(axes_final, final_tags):
            steps, values = get_scalars(event_file, tag)
            if steps:
                ax.bar(range(len(values)), values)
                ax.set_xlabel('Index')
                ax.set_ylabel(tag.replace('final/', ''))
                ax.set_title(tag)
                ax.grid(True, alpha=0.3)
        for ax in axes_final[len(final_tags):]:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'fold_{fold}_final_metrics.png'), dpi=120, bbox_inches='tight')
        plt.close()
    
    return True

def main():
    base_dir = '/home/shanyiye/CLAM/results'
    output_base = '/home/shanyiye/CLAM/analysis_results/training_curves'
    
    experiments = [
        'dlbcl_gcb_nongcb_clam_mb_nows_s1',
        'dlbcl_gcb_nongcb_clam_sb_nows_s1',
        'dlbcl_gcb_nongcb_mil_s1',
    ]
    
    model_names = {
        'dlbcl_gcb_nongcb_clam_mb_nows_s1': 'CLAM_MB_noWS',
        'dlbcl_gcb_nongcb_clam_sb_nows_s1': 'CLAM_SB_noWS',
        'dlbcl_gcb_nongcb_mil_s1': 'MIL',
    }
    
    for exp_name in experiments:
        display_name = model_names.get(exp_name, exp_name)
        output_dir = os.path.join(output_base, display_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nProcessing: {display_name}")
        
        success_count = 0
        for fold in range(10):
            print(f"  Fold {fold}...", end=" ")
            if plot_all_curves(exp_name, fold, output_dir):
                success_count += 1
                print("OK")
            else:
                print("Skipped")
        
        print(f"  Completed {success_count}/10 folds")

    print(f"\n\nAll training curves saved to: {output_base}")
    print("\nGenerated files:")
    for name in model_names.values():
        dir_path = os.path.join(output_base, name)
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            print(f"  {name}: {len(files)} files")

if __name__ == '__main__':
    main()
