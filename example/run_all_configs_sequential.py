import subprocess
import sys

def main():
    config_files = [
        'config/config_cifar10.yaml',
        'config/config_cifar100.yaml', 
        'config/config_cinic10.yaml',
        'config/config_svhn10.yaml',
        'config/config_tiny200.yaml'
    ]
    
    # Get command line arguments (excluding script name)
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    for i, config_file in enumerate(config_files, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(config_files)}] Starting training with config: {config_file}")
        print(f"{'='*80}")
        
        # Build command
        cmd = ['python', 'main.py'] + args + ['--c', config_file]
        
        try:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)
            print(f"✅ [{i}/{len(config_files)}] Completed training with config: {config_file}")
        except subprocess.CalledProcessError as e:
            print(f"❌ [{i}/{len(config_files)}] Error with config {config_file}: {e}")
            continue
        except KeyboardInterrupt:
            print(f"\n⚠️ Training interrupted by user")
            break
    
    print(f"\n🎉 All training completed!")

if __name__ == "__main__":
    main()