import torch
import os
from argparse import Namespace
from imbalanceddl.net.network import build_model
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.strategy.build_trainer import build_trainer
import yaml
import re
import ast
from types import SimpleNamespace
sub_dirs =['tiny200_exp_0.01_DRW_200_1598']
def reload_checkpoint(dir_path, argument_path):
    def parse_strategy(dir_path):
        base_name = os.path.basename(dir_path)
        parts = base_name.split('_')
        if len(parts) >= 4:
            strategy = parts[3]
            return strategy
        else:
            raise ValueError("Directory name does not contain enough parts to extract strategy.")

    def load_config(argument_path):
        """Load config from args.txt file"""
        with open(argument_path, "r") as f:
            content = f.read().strip()
        
        print(f"[DEBUG] Raw content: {content[:200]}...")  # Show first 200 chars
        
        # Remove outer braces { } if present
        if content.startswith("{") and content.endswith("}"):
            content = content[1:-1]

        # Remove Namespace( ... ) wrapper
        if content.startswith("Namespace(") and content.endswith(")"):
            content = content[len("Namespace("):-1]
        
        print(f"[DEBUG] After cleanup: {content[:200]}...")
        
        # Convert key=value pairs to 'key': value format
        # Handle different value types
        def convert_value(match):
            key = match.group(1)
            value = match.group(2).strip()
            
            # Handle None
            if value == 'None':
                return f"'{key}': None"
            # Handle True/False
            elif value in ['True', 'False']:
                return f"'{key}': {value}"
            # Handle strings (already quoted)
            elif value.startswith("'") and value.endswith("'"):
                return f"'{key}': {value}"
            elif value.startswith('"') and value.endswith('"'):
                return f"'{key}': {value}"
            # Handle numbers (int/float)
            elif re.match(r'^-?\d+(\.\d+)?([eE][+-]?\d+)?$', value):
                return f"'{key}': {value}"
            # Handle lists/tuples
            elif value.startswith('[') or value.startswith('('):
                return f"'{key}': {value}"
            # Default: treat as string
            else:
                return f"'{key}': '{value}'"
        
        # Convert key=value to 'key': value
        content = re.sub(r"(\w+)=([^,]+)", convert_value, content)
        
        # Wrap with braces to make it a valid dict
        dict_str = "{" + content + "}"
        
        print(f"[DEBUG] Final dict string: {dict_str[:200]}...")
        
        try:
            # Try to evaluate as dictionary
            config_dict = ast.literal_eval(dict_str)
            return SimpleNamespace(**config_dict)
        except (ValueError, SyntaxError) as e:
            print(f"❌ Error parsing with ast.literal_eval: {e}")
            print(f"Problematic string: {dict_str}")
            
            # Alternative: Manual parsing for problematic cases
            return manual_parse_args(content)
    
    def manual_parse_args(content):
        """Manual parsing as fallback"""
        config_dict = {}
        
        # Split by comma, but be careful with nested structures
        parts = []
        bracket_count = 0
        current_part = ""
        
        for char in content:
            if char in '[({':
                bracket_count += 1
            elif char in '])}':
                bracket_count -= 1
            elif char == ',' and bracket_count == 0:
                parts.append(current_part.strip())
                current_part = ""
                continue
            current_part += char
        
        if current_part.strip():
            parts.append(current_part.strip())
        
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Convert value types
                if value == 'None':
                    config_dict[key] = None
                elif value == 'True':
                    config_dict[key] = True
                elif value == 'False':
                    config_dict[key] = False
                elif value.startswith("'") and value.endswith("'"):
                    config_dict[key] = value[1:-1]  # Remove quotes
                elif value.startswith('"') and value.endswith('"'):
                    config_dict[key] = value[1:-1]  # Remove quotes
                else:
                    try:
                        # Try to convert to number
                        if '.' in value:
                            config_dict[key] = float(value)
                        else:
                            config_dict[key] = int(value)
                    except ValueError:
                        # Keep as string
                        config_dict[key] = value
        
        print(f"[DEBUG] Manually parsed config: {config_dict}")
        return SimpleNamespace(**config_dict)
    
    config = load_config(argument_path)
    strategy = parse_strategy(dir_path)
    config.strategy = strategy
    
    print('DEBUG STRATEGY', strategy)
    print('DEBUG CONFIG', vars(config))
    
    model = build_model(config)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=float(config.weight_decay)
    )
    print('[DEBUG] INITIALIZE MODEL AND OPTIMIZER')
    
    start_epoch = config.start_epoch
    checkpoint_loaded = False
    
    # Get actual files in the directory
    checkpoint_files = os.listdir(dir_path)
    print(f'[DEBUG] Files found in directory: {checkpoint_files}')
    
    for filename in checkpoint_files:
        if filename.endswith('ckpt.pth.tar'):
            checkpoint_path = os.path.join(dir_path, filename)
            print(f'[DEBUG] Loading checkpoint: {checkpoint_path}')
            
            try:
                checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(checkpoint_data['state_dict'])
                optimizer.load_state_dict(checkpoint_data['optimizer'])
                
                loaded_epoch = checkpoint_data['epoch']
                start_epoch = loaded_epoch + 1
                config.start_epoch = start_epoch
                checkpoint_loaded = True
                
                print(f"✅ Loaded Checkpoint '{filename}' (epoch {loaded_epoch})")
                print(f"🔄 Will resume training from epoch {start_epoch}")
                
                if 'best_acc1' in checkpoint_data:
                    print(f"📊 Best accuracy: {checkpoint_data['best_acc1']:.2f}")
                
                break
                
            except Exception as e:
                print(f'❌ Error loading checkpoint {filename}: {e}')
                continue
    
    if not checkpoint_loaded:
        print('⚠️ No valid checkpoint found - starting from scratch')
        print(f'📅 Starting from epoch {start_epoch}')
    
    print('[DEBUG] RELOADED MODEL AND OPTIMIZER FROM CHECKPOINT')
    return model, optimizer, config, start_epoch

def retrain_checkpoint(dir_path, config_path):
    model, optimizer, config, start_epoch = reload_checkpoint(dir_path, config_path)
    dataset_name = os.path.basename(dir_path).split('_')[0]
    print('🤖 Model:', type(model).__name__)
    print('⚙️ Optimizer:', type(optimizer).__name__)  
    print('📅 Start Epoch:', start_epoch)
    print('🔧 Config Start Epoch:', config.start_epoch)
    
    print(f"\n[DEBUG] CHECK UPDATED CONFIG START EPOCH: {config.start_epoch}")
    
    if hasattr(config, 'best_model') and config.best_model is not None:
        print("=> Eval with Best Model!")
        # trainer.eval_best_model()
    else:
        print("=> Start Train Val!")
        imbalance_dataset = ImbalancedDataset(config, dataset_name=dataset_name)
        trainer = build_trainer(config, imbalance_dataset, model=model, strategy=config.strategy)
        trainer.do_train_val()
    
    print("=> Checkpoint loading test completed!")
import os

def get_experiment_paths(sub_dir, base_checkpoint_dir=None, base_log_dir=None):
    """
    Generate checkpoint and config paths from subdirectory name
    
    Args:
        sub_dir: e.g., 'tiny200_exp_0.01_ERM_200_1916'
        base_checkpoint_dir: Base checkpoint directory (auto-detected if None)
        base_log_dir: Base log directory (auto-detected if None)
    
    Returns:
        tuple: (checkpoint_folder, config_path)
    """
    # Extract dataset from subdirectory name
    parts = sub_dir.split('_')
    dataset = parts[0]  # e.g., 'tiny200', 'cifar100', 'cifar10'
    
    # Auto-detect base directories if not provided
    if base_checkpoint_dir is None:
        if dataset == 'tiny200':
            base_checkpoint_dir = '/home/hamt/light_weight/imbalanced-DL/example/checkpoint_tiny200'
        elif dataset == 'cifar100':
            base_checkpoint_dir = '/home/hamt/light_weight/imbalanced-DL/example/checkpoint_cifar100'
        elif dataset == 'cifar10':
            base_checkpoint_dir = '/home/hamt/light_weight/imbalanced-DL/example/checkpoint_cifar10'
        else:
            base_checkpoint_dir = f'/home/hamt/light_weight/imbalanced-DL/example/checkpoint_{dataset}'
    
    if base_log_dir is None:
        if dataset == 'tiny200':
            base_log_dir = '/home/hamt/light_weight/imbalanced-DL/example/log_tiny200'
        elif dataset == 'cifar100':
            base_log_dir = '/home/hamt/light_weight/imbalanced-DL/example/log_cifar100'
        elif dataset == 'cifar10':
            base_log_dir = '/home/hamt/light_weight/imbalanced-DL/example/log_cifar10'
        else:
            base_log_dir = f'/home/hamt/light_weight/imbalanced-DL/example/log_{dataset}'
    
    # Build paths
    checkpoint_folder = os.path.join(base_checkpoint_dir, sub_dir)
    config_path = os.path.join(base_log_dir, sub_dir, 'args.txt')
    
    return checkpoint_folder, config_path

def main(sub_dirs):
    for sub_dir in sub_dirs:
        checkpoint_folder, config_path = get_experiment_paths(sub_dir)
        print(f"✅ {sub_dir}:\n   📁 Checkpoint: {checkpoint_folder}\n   📄 Config: {config_path}\n")
        retrain_checkpoint(checkpoint_folder, config_path)

if __name__ == "__main__":
    main(sub_dirs)