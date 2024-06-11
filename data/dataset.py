from argparse import ArgumentError
import torch
from torchvision import datasets
from torch.utils.data import Dataset

from .loader import pil_loader

import os
import pandas as pd

from torchvision.datasets import VisionDataset

from PIL import Image

from multi_level_split.util import train_test_split as multilevel_split



class CustomizedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=pil_loader):
        super(CustomizedImageFolder, self).__init__(root, transform, target_transform, loader=loader)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class DatasetFromDict(Dataset):
    def __init__(self, imgs, transform=None, loader=pil_loader):
        super(DatasetFromDict, self).__init__()
        self.imgs = imgs
        self.loader = loader
        self.transform = transform
        self.targets = [img[1] for img in imgs]
        self.classes = sorted(list(set(self.targets)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = self.loader(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, label


class KaggleDataset(datasets.VisionDataset):
    """Creates pytorch Dataset with Eyepacs fundus images and DR grades
    
    Selection of left, right or both eyes can be specified.
    
    """
    def __init__(self, root, split='train', transform=None, target_transform=None, loader=pil_loader):

        super(KaggleDataset, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        
        self.loader = loader
        
        if split == 'train':
            self.image_path = os.path.join(root, 'train')
            label_csv = os.path.join(root, 'trainLabels.csv')
            
            self.metadata = pd.read_csv(label_csv)
            
        elif split in ['val', 'test']:
            self.image_path = os.path.join(root, 'test')
            label_csv = os.path.join(root, 'retinopathy_solution.csv')
        
            usage_dict = {'val': 'Public', 'test': 'Private'}
            
            self.metadata = pd.read_csv(label_csv)

            self.metadata = self.metadata[self.metadata['Usage'] == usage_dict[split]].reset_index(drop=True)
        
        else:
            raise ArgumentError(f'Invalid split: {split}')   

        # TODO: remove images that are not there
        drop_indices = []
        
        for idx, row in self.metadata.iterrows():
        
            img_id = row['image']
            img_path = os.path.join(self.image_path, f"{img_id}.png")

            if not os.path.exists(img_path):
                drop_indices.append(idx)
            
        self.metadata = self.metadata.drop(drop_indices).reset_index(drop=True)
        

        self.targets = list(self.metadata['level'])
        self.classes = sorted(list(set(self.targets)))
        
           
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):   
        img_id = self.metadata['image'][idx]
        img_path = os.path.join(self.image_path, f"{img_id}.png")
        
        img = self.loader(img_path)
       
        if self.transform is not None:
            img = self.transform(img)

        label = self.metadata['level'][idx]

        return (img, label)

    # @property
    # def targets(self):
    #     return torch.tensor(self.metadata['level'].values)
    


class EyepacsDataset(VisionDataset):
        
    def __init__(self, data_root,
                 root='', 
                 transform=None, 
                 target_transform=None,
                 split='train'):
        super(EyepacsDataset, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
                
        self.image_path = os.path.join(data_root, 'data_raw', 'images')
        meta_csv = os.path.join(data_root, 'data_processed', 'metadata', 'metadata_image.csv')
        metadata_df = pd.read_csv(meta_csv)

        self._split_dict = {
            'train': 0,
            'test': 1,
            'val': 2
        }
        self._split_names = {
            'train': 'Train',
            'test': 'Test',
            'val': 'Validation',
        }

        if split not in self._split_dict:
            raise ArgumentError(f'split not recognised: {split}')

        dev, test = multilevel_split(metadata_df, 'image_id', 
                                    split_by='patient_id', 
                                    test_split=0.2,
                                    seed=12345)

        train, val = multilevel_split(dev, 'image_id', 
                                    split_by='patient_id', 
                                    test_split=0.25,
                                    seed=12345)

        data = {'train': train, 'val': val, 'test': test}

        self._metadata_df = data[split]

        # declutter: keep only images with the following characteristics
        sides = {'left': 1, 'right': 0}
        fields = {'field 1': 1, 'field 2': 2, 'field 3': 3}
        genders = {'Male': 0, 'Female': 1, 'Other': 2}
        image_qualities = {'Insufficient for Full Interpretation': 0, 'Adequate': 1, 'Good': 2, 'Excellent': 3}
        ethnicities = {'Latin American': 0, 'Caucasian': 1, 'African Descent': 2, 'Asian': 3, 'Indian subcontinent origin': 4,
                       'Native American': 5, 'Multi-racial': 6}

        # filter
        keep_fields = ['field 1']
        keep_quality = ['Adequate', 'Good','Excellent']
        
        self._metadata_df = self._metadata_df.query(f'image_side in {list(sides)}')
        self._metadata_df = self._metadata_df.query(f'image_field in {list(keep_fields)}')
        self._metadata_df = self._metadata_df.query(f'patient_gender in {list(genders)}')
        self._metadata_df = self._metadata_df.query(f'session_image_quality in {list(keep_quality)}')
        self._metadata_df = self._metadata_df.query(f'patient_ethnicity in {list(ethnicities)}')


        # Get the y values
        self._y_array = torch.CharTensor(self._metadata_df['diagnosis_image_dr_level'].values)
        self._n_classes = 5
        
        # Get filenames
        self._input_array = [os.path.join(self.image_path, ele) for ele in self._metadata_df['image_path'].values]

        self._side_array = torch.LongTensor([sides[ele] for ele in self._metadata_df['image_side']])
        self._field_array = torch.LongTensor([fields[ele] for ele in self._metadata_df['image_field']])
        self._gender_array = torch.LongTensor([genders[ele] for ele in self._metadata_df['patient_gender']])
        self._quality_array = torch.LongTensor([image_qualities[ele] for ele in self._metadata_df['session_image_quality']])
        self._ethnicity_array = torch.LongTensor([ethnicities[ele] for ele in self._metadata_df['patient_ethnicity']])

        self._metadata_array = torch.stack(
            (self._side_array,
             self._field_array,
             self._gender_array,
             self._quality_array,
             self._ethnicity_array,
             self._y_array,
             ),
            dim=1)
        self._metadata_fields = ['side', 'field', 'gender', 'quality', 'ethnicity', 'y']        
        
        
        self.targets = list(self._y_array)
        self.classes = sorted(list(set(self.targets)))
        

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        # Any transformations are handled by the WILDSSubset
        # since different subsets (e.g., train vs test) might have different transforms
        x = self.get_input(idx)
        y = self.targets[idx]
        
        return x, y

    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """

        img_filename = os.path.join(
            self.image_path,
            self._input_array[idx])
        
        with open(img_filename, 'rb') as f:
            x = Image.open(f).convert('RGB')
                
        if self.transform is not None:
            x = self.transform(x)
        
        return x

    @property
    def side(self):
        return self._side_array

    @property
    def field(self):
        return self._field_array

    @property
    def gender(self):
        return self._gender_array

    @property
    def quality(self):
        return self._quality_array

    @property
    def ethnicity(self):
        return self._ethnicity_array

    @property
    def y_array(self):
        """
        A Tensor of targets (e.g., labels for classification tasks),
        with y_array[i] representing the target of the i-th data point.
        y_array[i] can contain multiple elements.
        """
        return self._y_array

    @property
    def n_classes(self):
        """
        Number of classes for single-task classification datasets.
        Used for logging and to configure models to produce appropriately-sized output.
        None by default.
        Leave as None if not applicable (e.g., regression or multi-task classification).
        """
        return getattr(self, '_n_classes', None)

    @property
    def metadata_fields(self):
        """
        A list of strings naming each column of the metadata table, e.g., ['hospital', 'y'].
        Must include 'y'.
        """
        return self._metadata_fields

    @property
    def metadata_array(self):
        """
        A Tensor of metadata, with the i-th row representing the metadata associated with
        the i-th data point. The columns correspond to the metadata_fields defined above.
        """
        return self._metadata_array

