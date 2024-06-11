import os
import sys
import time
import random

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils.func import *
from train import train, evaluate
from utils.metrics import Estimator
from data.builder import generate_dataset
from modules.builder import generate_model


def main():
    # load conf and paths files
    args = parse_config()
    cfg, cfg_paths = load_config(args)   
    cfg.data_paths = data_path(cfg, cfg_paths)

    # Test
    cfg.base.test = True
    if cfg.base.test:
        print('########## test')
        cfg.base.test = True
        cfg.base.sample = 30
        cfg.train.epochs = 3
     
    # create folder
    cfg.save_paths = load_save_paths(cfg)
    save_path = cfg.save_paths.model 
    log_path = cfg.save_paths.logger
    print('save path ', save_path)
    print('logs path', log_path)
    
    if os.path.exists(save_path):
        pass
        warning = 'Save path {} exists.\nDo you want to overwrite it? (y/n)\n'.format(save_path)
        if not (args.overwrite or input(warning) == 'y'):
            sys.exit(0)
    else:
        os.makedirs(save_path)

    logger = SummaryWriter(log_path)

    copy_config(args.config, save_path)
    copy_config('./train.py', save_path)
    
    # print configuration
    if args.print_config:
        print_config({
            'BASE CONFIG': cfg.base,
            'DATA CONFIG': cfg.data,
            'TRAIN CONFIG': cfg.train
        })
    else:
        print_msg('LOADING CONFIG FILE: {}'.format(args.config))
    #'''

    # train
    set_random_seed(cfg.base.random_seed)
    cfg.base.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device', cfg.base.device)

    since = time.time()
    model = generate_model(cfg)
    train_dataset, test_dataset, val_dataset = generate_dataset(cfg)
    estimator = Estimator(cfg)
    #exit()
    #'''
    train(
        cfg=cfg,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        estimator=estimator,
        logger=logger
    )
    #'''

    ################## test ############
    ## Manual test 
    #save_path = './save_models'

    name = 'best_validation_weights'
    if cfg.data.binary: # only use for binary task to directly save the performance of the best model on the test set
        model_list = ['acc', 'auc', 'spec', 'sens', 'prec']
        model_name = ['Accuracy', 'AUC', 'Specificity', 'Sensitivity', 'Precision']
    else:
        model_list = ['acc', 'kappa'] #['acc'] 'loss'
        model_name = ['Accuracy', 'Kappa'] # ['Accuracy'] 'loss'
        
    for i in range(len(model_list)):
        print('========================================')
        print(f'This is the performance of the final model base on the best {model_name[i]}')
        checkpoint = os.path.join(save_path, f'{name}_{model_list[i]}.pt')
        evaluate(cfg, model, checkpoint, val_dataset, estimator, type_ds='validation')
        
        print('')
        evaluate(cfg, model, checkpoint, test_dataset, estimator, type_ds='test')
        print('')

    time_elapsed = time.time() - since
    print('Training and evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    main()
