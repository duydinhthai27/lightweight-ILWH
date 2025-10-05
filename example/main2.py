import multiprocessing as mp
import os, time, numpy as np, signal, sys, argparse, atexit, gc
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
selected_gpu = None

def cleanup_at_exit():
    """Final cleanup function called at program exit"""
    try:
        # Force cleanup of any remaining multiprocessing resources
        for child in mp.active_children():
            child.terminate()
            child.join(timeout=1)
            try:
                child.close()
            except:
                pass
        gc.collect()
    except:
        pass

# Register cleanup function to run at exit
atexit.register(cleanup_at_exit)

def signal_handler(signum, frame):
    """Handle graceful shutdown on SIGINT/SIGTERM"""
    global shutdown_requested
    shutdown_requested = True
    print(f"\n[SHUTDOWN] Received signal {signum}. Gracefully shutting down...")
    
def cleanup_dead_processes(processes):
    """Clean up dead processes and return count of cleaned up processes"""
    cleaned_count = 0
    dead_pids = []
    
    for pid, proc in list(processes.items()):
        if not proc.is_alive():
            try:
                proc.join(timeout=2)  # Wait up to 2 seconds for cleanup
                proc.close()  # Explicitly close the process to free resources
                dead_pids.append(pid)
                cleaned_count += 1
            except Exception as e:
                print(f"Error cleaning up process {pid}: {e}")
                # Try to close anyway to prevent resource leaks
                try:
                    proc.close()
                except:
                    pass
                dead_pids.append(pid)  # Remove it anyway to prevent accumulation
    
    # Remove dead processes from tracking
    for pid in dead_pids:
        processes.pop(pid, None)
    
    if cleaned_count > 0:
        print(f"Cleaned up {cleaned_count} dead processes")
    
    return cleaned_count

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


def run_single_training(task, queue, gpu_id, base_config):
    """Worker: run training and send results via queue"""
    strategy, augmentation, dataset_name, round_id = task
    pid = os.getpid()
    
    # Get sampling name from config and create short version for process title
    sampling = getattr(base_config, 'sampling', 'WeightedFixedBatchSampler')  # Default fallback
    sampling_short = get_sampling_short_name(sampling)
    
    # Set process title for easy identification
    proc_title = f"{dataset_name}-{strategy}-{augmentation}-{sampling_short}-r{round_id}"
    setproctitle(proc_title)
    
    # CRITICAL: Set GPU device BEFORE importing torch or any torch-related modules
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"PID {pid}: Set CUDA_VISIBLE_DEVICES to {gpu_id}")
    print(f"PID {pid}: CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"PID {pid}: Process title set to: {proc_title}")
    
    # Import ALL torch-related modules AFTER setting environment variable
    import torch
    from imbalanceddl.utils import fix_all_seed, prepare_store_name, prepare_folders
    from imbalanceddl.net.network import build_model
    from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
    from imbalanceddl.strategy.build_trainer import build_trainer
    
    print(f"PID {pid}: Started training {strategy}+{augmentation} round {round_id}")
    print(f"PID {pid}: Available CUDA devices after restriction: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"PID {pid}: Current CUDA device: {torch.cuda.current_device()}")
        for i in range(torch.cuda.device_count()):
            print(f"PID {pid}: Device {i}: {torch.cuda.get_device_name(i)}")

    try:
         # Use base config instead of re-parsing args to avoid conflicts
        config = base_config
        config.strategy = strategy
        config.data_augment = augmentation
        config.dataset = dataset_name
        # Ensure sampling is set (use existing value or default)
        if not hasattr(config, 'sampling') or config.sampling is None:
            config.sampling = 'WeightedFixedBatchSampler'  # Default value
        
        # CRITICAL: After setting CUDA_VISIBLE_DEVICES, the GPU should be referenced as device 0
        config.gpu = 0  # Override the original GPU ID to use device 0 in the restricted environment
        # print(f"PID {pid}: Set config.gpu to 0 (remapped from original GPU {gpu_id})")

        # Unique seed
        SEED = (config.seed + round_id * 1000 + hash(strategy + augmentation) % 1000
                if config.seed is not None
                else np.random.randint(10000) + round_id * 1000)
        config.seed = SEED

        prepare_store_name(config)
        prepare_folders(config)
        fix_all_seed(config.seed)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        model = build_model(config)
        imbalance_dataset = ImbalancedDataset(config, dataset_name=config.dataset)
        trainer = build_trainer(config, imbalance_dataset, model=model, strategy=config.strategy)

        if config.best_model is not None:
            trainer.eval_best_model()
        else:
            if config.strategy == "M2m":
                trainer.do_train_val_m2m()
            else:
                trainer.do_train_val()

        # Cleanup
        del model, imbalance_dataset, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        msg = f"[OK] PID {pid} finished Round {round_id} - {strategy}+{augmentation}+{sampling}"
        queue.put(("success", task, msg))

    except Exception as e:
        sampling = getattr(base_config, 'sampling', 'WeightedFixedBatchSampler')
        msg = f"[ERROR] PID {pid} failed Round {round_id} - {strategy}+{augmentation}+{sampling}: {str(e)}"
        queue.put(("error", task, msg))
    finally:
        # Final cleanup attempt
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass


def create_all_tasks(selected_strategy, config):
    dataset_name = config.dataset
    
    # Customizable augmentation runs - modify this dictionary as needed
    # Format: {'augmentation_name': number_of_runs}
    augmentation_runs = {
        'randaugment': 0,
        'cutout': 0,
        'None': 0,
        'autoaugment_cifar10': 3  # Will be adjusted based on dataset
    }
    
    # Adjust autoaugment based on dataset (keeping original logic)
    if dataset_name in ["cifar100", "cinic10", "cifar10"]:
        augmentation_runs['autoaugment_cifar10'] = augmentation_runs.get('autoaugment_cifar10', 3)
        # Remove other autoaugment types if they exist
        augmentation_runs.pop('autoaugment_imagenet', None)
        augmentation_runs.pop('autoaugment_svhn', None)
        augmentation_runs.pop('autoaugment', None)
    elif dataset_name == "tiny200":
        augmentation_runs['autoaugment_imagenet'] = augmentation_runs.get('autoaugment_imagenet', 3)
        augmentation_runs.pop('autoaugment_cifar10', None)
        augmentation_runs.pop('autoaugment_svhn', None) 
        augmentation_runs.pop('autoaugment', None)
    elif dataset_name == "svhn":
        augmentation_runs['autoaugment_svhn'] = augmentation_runs.get('autoaugment_svhn', 3)
        augmentation_runs.pop('autoaugment_cifar10', None)
        augmentation_runs.pop('autoaugment_imagenet', None)
        augmentation_runs.pop('autoaugment', None)
    else:
        augmentation_runs['autoaugment'] = augmentation_runs.get('autoaugment', 3)
        augmentation_runs.pop('autoaugment_cifar10', None)
        augmentation_runs.pop('autoaugment_imagenet', None)
        augmentation_runs.pop('autoaugment_svhn', None)

    all_tasks = []
    for augmentation, num_runs in augmentation_runs.items():
        for round_id in range(num_runs):
            all_tasks.append((selected_strategy, augmentation, dataset_name, round_id))
    
    return all_tasks, augmentation_runs


def main():
    # Set main process title
    setproctitle("main2-manager")
    
    # Import get_args here to avoid early torch initialization
    from imbalanceddl.utils.config import get_args
    
    # Get configuration from utils
    config = get_args()
    
    # Check for required parameters
    if not hasattr(config, 'gpu') or config.gpu is None:
        print("ERROR: --gpu parameter is required")
        print("Usage: python main2.py --gpu <gpu_id> --strategy <strategy>")
        sys.exit(1)
    
    if not hasattr(config, 'strategy') or config.strategy is None:
        print("ERROR: --strategy parameter is required")
        print(f"Available strategies: {strategies}")
        print("Usage: python main2.py --gpu <gpu_id> --strategy <strategy>")
        sys.exit(1)
    
    if config.strategy not in strategies:
        print(f"ERROR: Invalid strategy '{config.strategy}'")
        print(f"Available strategies: {strategies}")
        sys.exit(1)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    # Note: SIGKILL cannot be caught or handled - it immediately terminates the process
    # But we can try to set it up (will raise OSError on most systems)
    try:
        signal.signal(signal.SIGKILL, signal_handler)
    except OSError:
        # SIGKILL cannot be caught, this is expected behavior
        pass
    
    print("=== TRAINING SESSION STARTED ===")
    print(f"GPU: {config.gpu}")
    print(f"Strategy: {config.strategy}")
    sampling = getattr(config, 'sampling', 'WeightedFixedBatchSampler')
    sampling_short = get_sampling_short_name(sampling)
    print(f"Sampling: {sampling} ({sampling_short})")
    dataset_name = config.dataset
    # Update manager process title with sampling info
    setproctitle(f"main2-manager-{dataset_name}-{config.strategy}-{sampling_short}")
    
    all_tasks, augmentation_runs = create_all_tasks(config.strategy, config)
    print(f"Total tasks: {len(all_tasks)}")
    print("Augmentation runs:")
    for aug, runs in augmentation_runs.items():
        print(f"  {aug}: {runs} runs")

    max_workers = 3
    queue = mp.Queue()
    processes = {}   # pid -> (process, task) tuple
    completed, errors = 0, 0
    task_iter = iter(all_tasks)
    last_status_time = time.time()

    def launch_next_task():
        """Launch the next available task if we have capacity"""
        if len(processes) < max_workers:
            try:
                task = next(task_iter)
                # Create a copy of config to avoid shared state issues
                import copy
                config_copy = copy.deepcopy(config)
                p = mp.Process(target=run_single_training, args=(task, queue, config.gpu, config_copy))
                p.start()
                processes[p.pid] = (p, task)
                print(f"[LAUNCH] PID {p.pid} started for task {task} (active: {len(processes)})")
                return True
            except StopIteration:
                return False
        return False

    # Launch initial batch
    for i in range(min(max_workers, len(all_tasks))):
        if launch_next_task():
            if i < max_workers - 1:  # Don't sleep after the last launch
                time.sleep(3)  # 3 second delay between launches
    
    while processes or (completed + errors < len(all_tasks)):
        # Check for shutdown signal
        if shutdown_requested:
            print("Shutdown requested, terminating remaining processes...")
            for pid, (proc, task) in list(processes.items()):
                if proc.is_alive():
                    proc.terminate()
                    print(f"Terminated process {pid}")
                try:
                    proc.join(timeout=2)
                    proc.close()
                except Exception as e:
                    print(f"Error during shutdown cleanup of {pid}: {e}")
            # Clear the processes dict
            processes.clear()
            break

        # Process results
        try:
            status, task, msg = queue.get(timeout=1)
            if status == "success":
                completed += 1
            elif status == "error":
                errors += 1
            print(f"Progress: {completed + errors}/{len(all_tasks)} - {msg}")
        except Exception:
            # No result in timeout -> continue
            pass
        
        # Clean up finished processes and launch new ones
        finished_pids = []
        for pid, (proc, task) in list(processes.items()):
            if not proc.is_alive():
                try:
                    proc.join(timeout=0.1)
                    proc.close()
                except:
                    pass
                finished_pids.append(pid)
        
        # Remove finished processes
        for pid in finished_pids:
            processes.pop(pid, None)
        
        # Launch new tasks to fill capacity
        if finished_pids:
            for i in range(len(finished_pids)):
                if launch_next_task():
                    if i < len(finished_pids) - 1:  # Don't sleep after the last launch
                        time.sleep(3)  # 3 second delay between launches
        
        # Status update (every 60 seconds)
        current_time = time.time()
        if current_time - last_status_time > 60:
            print(f"Status: {len(processes)} active, {completed} completed, {errors} errors")
            last_status_time = current_time

        # Stop when all tasks completed and no active processes
        if completed + errors >= len(all_tasks) and len(processes) == 0:
            break

    # Final cleanup - properly close all processes and resources
    for pid, (proc, task) in list(processes.items()):
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()  # Force kill if still alive
        try:
            proc.close()  # Explicitly close to free resources
        except:
            pass
    
    # Close and join the queue to prevent resource leaks
    try:
        queue.close()
        queue.join_thread()
    except Exception as e:
        print(f"Queue cleanup error: {e}")
    
    # Force garbage collection to help clean up any remaining resources
    import gc
    gc.collect()
    
    # Additional multiprocessing cleanup
    try:
        # Clean up any remaining multiprocessing resources
        mp.active_children()  # This helps clean up zombie processes
    except:
        pass

    print("="*60)
    print(f"ALL DONE: {completed} completed, {errors} errors")
    print("="*60)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()