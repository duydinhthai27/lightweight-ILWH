import numpy as np
import torch

from imbalanceddl.utils import fix_all_seed, prepare_store_name, prepare_folders
from imbalanceddl.net.network import build_model
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.strategy.build_trainer import build_trainer
from imbalanceddl.utils.config import get_args

def main():
    config = get_args()
    # Prepare Log
    prepare_store_name(config)
    print("=> Store Name = {}".format(config.store_name))
    prepare_folders(config)

    # Fix Seed
    if config.seed is not None:
        SEED = config.seed
    else:
        SEED = np.random.randint(10000)
        config.seed = SEED
    fix_all_seed(config.seed)

    print(torch.__version__)
    torch.cuda.empty_cache()

    if config.strategy == 'M2m':
        # Build Model
        model = build_model(config)

        # Build Dataset
        imbalance_dataset = ImbalancedDataset(config, dataset_name=config.dataset)
        # Build Trainer
        trainer = build_trainer(config,
                                imbalance_dataset,
                                model=model,
                                strategy=config.strategy)
    # Check M2m strategy
        if config.best_model is not None:
            print("=> Eval with Best Model !")
            trainer.eval_best_model()
        else:
            print("=> Start Train Val !")
            trainer.do_train_val_m2m()
        print("=> All Completed !")
    else:
        model = build_model(config)
        # Build Dataset
        imbalance_dataset = ImbalancedDataset(config, dataset_name=config.dataset)
        # Build Trainer
        trainer = build_trainer(config,
                                imbalance_dataset,
                                model=model,
                                strategy=config.strategy)
        # Test with Best Model or Train from scratch
        if config.best_model is not None:
            print("=> Eval with Best Model !")
            trainer.eval_best_model()
        else:
            print("=> Start Train Val !")
            trainer.do_train_val()
        print("=> All Completed !")

"""Kim modified this code to allow training with all strategies in a loop.
This will iterate through all strategies defined in the `strategiesMethod` list,
build the model, dataset, and trainer for each strategy, and execute the training process.
"""

def training_model(path=None):
    #config_path=r'config/config_cifar10.yaml'
    strategiesMethod = ['ERM', 'DRW', 'LDAM_DRW', 'Mixup_DRW', 'Remix_DRW','Reweight_CB', 'MAMix_DRW', 'M2m', 'DeepSMOTE']
    config = get_args()
    for strategy in strategiesMethod:
        print(f"Training with strategy: {strategy}")
        config.strategy = strategy
        #config.config = config_path
        #Prepare log
        prepare_store_name(config)
        print("=> Store Name = {}".format(config.store_name))
        prepare_folders(config)
        # Fix Seed
        if config.seed is not None:
            SEED = config.seed
        else:
            SEED = np.random.randint(10000)
            config.seed = SEED
        fix_all_seed(config.seed)
        print(config)
        torch.cuda.empty_cache()
        try:
            model=build_model(config)
            imbalance_dataset = ImbalancedDataset(config, dataset_name=config.dataset)
            trainer = build_trainer(config, imbalance_dataset, model=model, strategy=strategy)
            if config.best_model is not None:
                print("=> Eval with Best Model !")
                trainer.eval_best_model()
            else:
                print("=> Start Train Val !")
                trainer.do_train_val()
            print("=> Completed Training with Strategy: {}".format(strategy))
        except Exception as e:
            print(f"Error occurred during training with strategy {strategy}: {e}")


if __name__ == "__main__":
    # training_model()
    main()
