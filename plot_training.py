import os
import struct
import numpy as np
import matplotlib.pyplot as plt

def read_tfevents(filepath):
    """Read tensorflow event files (tensorboardX format)"""
    events = []
    with open(filepath, 'rb') as f:
        while True:
            try:
                header = f.read(8)
                if not header:
                    break
                encoded_len = struct.unpack('Q', header)[0]
                event_str = f.read(encoded_len)
                events.append(event_str)
            except:
                break
    return events

def parse_tfevents(filepath):
    """Parse tensorboard events and extract scalar data"""
    try:
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(filepath)
        ea.Reload()
        return ea
    except ImportError:
        print("tensorboard not installed, trying manual parsing...")
        return None

def get_scalars(event_file, tag):
    """Extract scalar values for a given tag"""
    ea = parse_tfevents(event_file)
    if ea:
        try:
            scalars = ea.Scalars(tag)
            steps = [s.step for s in scalars]
            values = [s.value for s in scalars]
            return steps, values
        except:
            return [], []
    return [], []

def main():
    base_dir = '/home/shanyiye/CLAM/results'
    exp_name = 'dlbcl_gcb_nongcb_clam_mb_nows_s1'
    fold = '0'
    
    event_file = os.path.join(base_dir, exp_name, fold, 
                             [f for f in os.listdir(os.path.join(base_dir, exp_name, fold)) if f.startswith('events')][0])
    
    print(f"Reading: {event_file}")
    
    tags = ['train/loss', 'train/class_0_acc', 'val/loss', 'val/auc', 'val/error']
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'Training Curves - {exp_name} (Fold {fold})', fontsize=14)
    
    data_found = False
    for i, tag in enumerate(tags):
        ax = axes[i // 2, i % 2]
        steps, values = get_scalars(event_file, tag)
        
        if steps:
            data_found = True
            ax.plot(steps, values, marker='o', markersize=3)
            ax.set_xlabel('Step')
            ax.set_ylabel(tag)
            ax.set_title(tag)
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f'{tag} (No data)')
    
    plt.tight_layout()
    output_file = f'training_curves_{exp_name}_fold{fold}.png'
    plt.savefig(output_file)
    print(f"Saved to: {output_file}")
    
    if not data_found:
        print("\nNo scalar data found. Checking available tags...")
        ea = parse_tfevents(event_file)
        if ea:
            print("Available tags:", ea.Tags()['scalars'])

if __name__ == '__main__':
    main()
