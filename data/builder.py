from argparse import ArgumentError
import os
import pickle

from torchvision import datasets

from .loader import pil_loader
from .transforms import data_transforms, simple_transform, our_transform
from .dataset import DatasetFromDict, CustomizedImageFolder, KaggleDataset, EyepacsDataset
from utils.func import mean_and_std, print_dataset_info
from .our_dataset import FundusDataset


def generate_dataset(cfg):
    
    if cfg.data.mean == 'auto' or cfg.data.std == 'auto':
        data_path = os.path.join(cfg.data_paths.root, cfg.data_paths.dset_dir)
            
        mean, std = auto_statistics(
            data_path, #cfg.base.data_path
            cfg.base.data_index,
            cfg.data.input_size,
            cfg.train.batch_size,
            cfg.train.num_workers,
            cfg
        )        
        cfg.data.mean = mean  
        cfg.data.std = std     

    if cfg.data.augmentation == 'other':
        train_transform, test_transform = our_transform(cfg)
    else:
        train_transform, test_transform = data_transforms(cfg)

    if 'kaggle' in cfg.base.dataset:
        datasets = generate_our_dataset_kaggle(
            cfg,
            train_transform,
            test_transform
        )
    elif cfg.base.dataset == 'eyepacs':
        datasets = generate_dataset_eyepacs(
            cfg.base.data_path,
            train_transform,
            test_transform
        )
    else:
        raise ArgumentError(f'Dataset not implemented: {cfg.base.dataset}')

    print_dataset_info(datasets)
    return datasets


def auto_statistics(data_path, data_index, input_size, batch_size, num_workers, cfg):
    print('Calculating mean and std of training set for data normalization.')
    transform = simple_transform(input_size)

    if data_index not in [None, 'None']:
        train_set = pickle.load(open(data_index, 'rb'))['train']
        train_dataset = DatasetFromDict(train_set, transform=transform)
    else:
        ''' generate the training dataset'''
        #os.path.join(data_path, 'train') # initial 
        #train_dataset = datasets.ImageFolder(train_path, transform=transform)
        train_dataset = FundusDataset(cfg, transform=transform)

    return mean_and_std(train_dataset, batch_size, num_workers)


def generate_dataset_from_folder(data_path, train_transform, test_transform):
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    val_path = os.path.join(data_path, 'val')

    train_dataset = CustomizedImageFolder(train_path, train_transform, loader=pil_loader)
    test_dataset = CustomizedImageFolder(test_path, test_transform, loader=pil_loader)
    val_dataset = CustomizedImageFolder(val_path, test_transform, loader=pil_loader)

    return train_dataset, test_dataset, val_dataset


def generate_dataset_from_pickle(pkl, train_transform, test_transform):
    data = pickle.load(open(pkl, 'rb'))
    train_set, test_set, val_set = data['train'], data['test'], data['val']

    train_dataset = DatasetFromDict(train_set, train_transform, loader=pil_loader)
    test_dataset = DatasetFromDict(test_set, test_transform, loader=pil_loader)
    val_dataset = DatasetFromDict(val_set, test_transform, loader=pil_loader)

    return train_dataset, test_dataset, val_dataset

def generate_dataset_kaggle(root, train_transform, test_transform):
        
    dset_train = KaggleDataset(root, split='train', transform=train_transform, loader=pil_loader)
    dset_val = KaggleDataset(root, split='val', transform=test_transform, loader=pil_loader)
    dset_test = KaggleDataset(root, split='test', transform=test_transform, loader=pil_loader)

    return dset_train, dset_test, dset_val

def generate_dataset_eyepacs(root, train_transform, test_transform):
                
    dset_train = EyepacsDataset(root, split='train', transform=train_transform)
    dset_val = EyepacsDataset(root, split='val', transform=test_transform)
    dset_test = EyepacsDataset(root, split='test', transform=test_transform)

    return dset_train, dset_test, dset_val

def generate_our_dataset_kaggle(cfg, train_transform, test_transform):
                
    dset_train = FundusDataset(cfg, transform=train_transform)
    dset_val = FundusDataset(cfg, train=False, transform=test_transform)
    dset_test = FundusDataset(cfg, train=False, test=True, transform=test_transform)

    return dset_train, dset_test, dset_val