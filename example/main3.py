import os
import time
import numpy as np
import signal
import sys
import argparse
from datetime import datetime

try:
    from setproctitle import setproctitle
except ImportError:
    # Fallback if setproctitle is not installed
    def setproctitle(title):
        pass

strategies = ['ERM', 'DRW', 'Mixup', 'Mixup_DRW']
sampling_options = ['Random', 'WeightedFixedBatchSampler', 'WeightedRandomBatchSampler']

def get_sampling_short_name(sampling):
    """Get shortened sampling name for process titles"""
    if sampling == 'WeightedFixedBatchSampler':
        return 'WFBS'
    elif sampling == 'WeightedRandomBatchSampler':
        return 'WRBS'
    elif sampling == 'Random':
        return 'Random'
    else:
        return sampling

# Global variables for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle graceful shutdown on SIGINT/SIGTERM"""
    global shutdown_requested
    shutdown_requested = True
    print(f"\n[SHUTDOWN] Received signal {signum}. Gracefully shutting down...")
    sys.exit(0)

def ListAug(datasetName):
    baseList = ['randaugment', 'cutout', 'None']
    if datasetName in ["cifar100", "cinic10", "cifar10"]:
        baseList.append('autoaugment_cifar10')
    elif datasetName == "tiny200":
        baseList.append('autoaugment_imagenet')
    elif datasetName == "svhn":
        baseList.append('autoaugment_svhn')
    else:
        baseList.append('autoaugment')
    return baseList

def run_single_training(strategy, sampling, augmentation, dataset_name, run_id, config):
    """Run a single training configuration"""
    
    # Set process title for easy identification
    sampling_short = get_sampling_short_name(sampling)
    proc_title = f"{dataset_name}-{strategy}-{sampling_short}-{augmentation}-r{run_id}"
    setproctitle(proc_title)
    
    print(f"\n{'='*80}")
    print(f"Starting: Strategy={strategy}, Sampling={sampling}, Augmentation={augmentation}, Run={run_id}")
    print(f"Process title: {proc_title}")
    print(f"{'='*80}")
    
    # Import torch-related modules when needed
    import torch
    from imbalanceddl.utils import fix_all_seed, prepare_store_name, prepare_folders
    from imbalanceddl.net.network import build_model
    from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
    from imbalanceddl.strategy.build_trainer import build_trainer
    
    try:
        # Configure the training
        config.strategy = strategy
        config.sampling = sampling
        config.data_augment = augmentation
        config.dataset = dataset_name
        
        # Unique seed for each run
        SEED = (config.seed + run_id * 1000 + hash(strategy + sampling + augmentation) % 1000
                if config.seed is not None
                else np.random.randint(10000) + run_id * 1000)
        config.seed = SEED

        prepare_store_name(config)
        prepare_folders(config)
        fix_all_seed(config.seed)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"GPU {config.gpu}: {torch.cuda.get_device_name(config.gpu)}")

        print(f"Building model and dataset...")
        model = build_model(config)
        imbalance_dataset = ImbalancedDataset(config, dataset_name=config.dataset)
        trainer = build_trainer(config, imbalance_dataset, model=model, strategy=config.strategy)

        print(f"Starting training...")
        start_time = time.time()
        
        if config.best_model is not None:
            trainer.eval_best_model()
        else:
            if config.strategy == "M2m":
                trainer.do_train_val_m2m()
            else:
                trainer.do_train_val()

        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{'='*80}")
        print(f"✓ COMPLETED: {strategy}+{sampling}+{augmentation} (Run {run_id})")
        print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"{'='*80}")

        # Cleanup
        del model, imbalance_dataset, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return True

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"✗ ERROR: {strategy}+{sampling}+{augmentation} (Run {run_id})")
        print(f"Error: {str(e)}")
        print(f"{'='*80}")
        
        # Cleanup on error
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        return False

def create_all_combinations(selected_strategy, config):
    """Create all combinations of strategy, sampling, and augmentation"""
    dataset_name = config.dataset
    
    # Get available augmentations for the dataset
    available_augmentations = ListAug(dataset_name)
    
    # Use the selected strategy and all sampling options
    combinations = []
    
    for sampling in sampling_options:
        for augmentation in available_augmentations:
            # Each combination runs twice (run_id 0 and 1)
            for run_id in range(2):
                combinations.append((selected_strategy, sampling, augmentation, dataset_name, run_id))
    
    return combinations

def main():
    # Set main process title
    setproctitle("main3-sequential")
    
    # Import get_args here to avoid early torch initialization
    from imbalanceddl.utils.config import get_args
    
    # Get configuration from utils
    config = get_args()
    
    # Check for required parameters
    if not hasattr(config, 'gpu') or config.gpu is None:
        print("ERROR: --gpu parameter is required")
        print("Usage: python main3.py --gpu <gpu_id> --strategy <strategy>")
        sys.exit(1)
    
    if not hasattr(config, 'strategy') or config.strategy is None:
        print("ERROR: --strategy parameter is required")
        print(f"Available strategies: {strategies}")
        print("Usage: python main3.py --gpu <gpu_id> --strategy <strategy>")
        sys.exit(1)
    
    if config.strategy not in strategies:
        print(f"ERROR: Invalid strategy '{config.strategy}'")
        print(f"Available strategies: {strategies}")
        sys.exit(1)
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=== SEQUENTIAL TRAINING SESSION STARTED ===")
    print(f"GPU: {config.gpu}")
    print(f"Strategy: {config.strategy}")
    print(f"Dataset: {config.dataset}")
    print(f"Available sampling methods: {sampling_options}")
    print(f"Available augmentations: {ListAug(config.dataset)}")
    
    # Set environment variable for GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
    print(f"Set CUDA_VISIBLE_DEVICES to {config.gpu}")
    
    # Create all combinations
    all_combinations = create_all_combinations(config.strategy, config)
    total_combinations = len(all_combinations)
    
    print(f"\nTotal combinations to run: {total_combinations}")
    print("Each combination will run twice (2 runs per combination)")
    
    # Run each combination sequentially
    completed = 0
    errors = 0
    start_time = time.time()
    
    for i, (strategy, sampling, augmentation, dataset_name, run_id) in enumerate(all_combinations, 1):
        if shutdown_requested:
            print("Shutdown requested, stopping...")
            break
        
        print(f"\n[{i}/{total_combinations}] Running combination:")
        print(f"  Strategy: {strategy}")
        print(f"  Sampling: {sampling}")
        print(f"  Augmentation: {augmentation}")
        print(f"  Run: {run_id + 1}/2")
        
        success = run_single_training(strategy, sampling, augmentation, dataset_name, run_id, config)
        
        if success:
            completed += 1
        else:
            errors += 1
        
        print(f"\nProgress: {i}/{total_combinations} combinations attempted")
        print(f"Success: {completed}, Errors: {errors}")
        
        # Brief pause between runs to allow system cleanup
        if i < total_combinations:  # Don't pause after the last run
            print("Pausing 5 seconds before next combination...")
            time.sleep(5)

    end_time = time.time()
    total_duration = end_time - start_time
    
    print("\n" + "="*80)
    print("=== TRAINING SESSION COMPLETED ===")
    print(f"Total combinations: {total_combinations}")
    print(f"Successful runs: {completed}")
    print(f"Failed runs: {errors}")
    print(f"Total time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    if completed > 0:
        print(f"Average time per successful run: {total_duration/completed:.2f} seconds")
    print("="*80)

if __name__ == "__main__":
    main()
