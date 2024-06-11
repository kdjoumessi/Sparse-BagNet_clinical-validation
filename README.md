# Sparse-BagNet: Clinical validation for early Diabetic Retinopathy Detection
This repository contains the official implementation of the [Sparse-BagNet](https://openreview.net/forum?id=us8BFTsWOq) for early Diabetic Retinopathy detection from the paper [An Inherently Interpretable AI model improves Screening Speed and Accuracy
for Early Diabetic Retinopathy](...) submitted at xxx by xxx.

## Model's architecture
![Model's architecture](./files/model_architecture.png)

## Development dataset
![Dev dataset](./files/dev_dataset.png)


## Dependencies
All packages required for running the code in the repository are listed in the file _requirements.txt_

## Data \& Preprocessing
### Data
The code in this repository uses publicly available Kaggle dataset for the [diabetic retinopathy detection challenge](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)

### Preprocessing
Each image was preprocessed by tightly cropping the circular mask of the retinal fundus and resize to 512 x 512. The code is available here: [preprocessing](https://github.com/berenslab/fundus\_circle\_cropping)

An ensemble of EfficientNets trained on the [ISBI2020 challenge dataset](https://isbi.deepdr.org/challenge2.html) was used to filter out images with low qualities. The resulting dataset (csv files) used for model training and evaluation are as follows: 
- [`training csv file`](./files/csv_files/kaggle_gradable_train.csv)
- [`validation csv file`](./files/csv_files/kaggle_gradable_val.csv)
- [`test csv file`](./files/csv_files/kaggle_gradable_test.csv) 

The image names used for figures are provided in [`images.txt`](./files/image.txt)

## How to use: training
### 1. Organize the dataset as follows:
```
├── main_folder
    ├── Kaggle_data
        ├── Images
        ├── kaggle_gradable_train.csv
        ├── kaggle_gradable_test.csv
        ├── kaggle_gradable_val.csv 
    ├── Outputs
    ├── configs
    ├── data
    ├── files
    ├── utils
    ├── modules  
    ├── main.py
    ├── train.py
```

Adjust paths to dataset in `configs/default.yaml`. Replace the value of
- `paths.root` with path to `xx`
- `paths.dset_dir` with path to `xx`
- `paths.save` with path to `xx` to save the model output (logs and model weight)
- `paths.model_dir` with path to `xx`

### 2. Update the training configurations and hyperparameters 
All experiments are fully specified by the configuration file located at `./configs/default.yaml`.

The training configurations including hyperparameters turning can be done in the main config file.

### 3. Run to train
- Create a virtual environment and install dependencies 
```shell
$ pip install requirements.txt
```
- Run a model with previously defined parameters
```shell
$ python main.py
```

### 4. Monitor the training step 
Monitor the training progress in website [127.0.0.1:6006](127.0.0.1:6006) by running:

```
$ tensorborad --logdir=/path/to/your/log --port=6006
```

## Reproducibility
### Figures and annotations
- Code for figures are available

### User study
- CSV files of images used for the grading and the annotation tasks are available in `./user_study/csv`
- Annotations masks from clinicians used in the user study to evaluate the ability of the model to localize DR-related lesions are available in `./user_study` 
- The data collected from the user study (including the decision time, confidence, and grade) are available in `./user_study/results` 

### Models's weights
The final models with the best validation weights used for all the experiments are as follows:
- [ResNet model](https://drive.google.com/file/d/19uxCKAGI7B29tL0C89ZRnUaSSfjf01wp/view?usp=drive_link)
- [sparse-BagNet model](https://drive.google.com/file/d/1-BlykANm7bhJytlg25laWIeLZAtihYUU/view?usp=drive_link)

## Acknowledge
-  This repository contains modified source code from [kdjoumessi/interpretable-sparse-activation](https://github.com/kdjoumessi/interpretable-sparse-activation) 

## Citation
```
  @inproceedings{donteu2023sparse,
  title={An Inherently Interpretable AI model improves Screening Speed and Accuracy for Early Diabetic Retinopathy},
  author={xx},
  booktitle={xx},
  year={2024}
}
```

Donteu Kerol R. Djoumessi., Indu Ilanchezian, Laura Kuhlewein, Hanna Faber., Christian Baumgartner, Bubacarr Bah, Philipp Berens, Lisa Koch. Sparse activations for interpretable disease grading. In Medical Imaging with Deep Learning, Nashville, United States, July 2023.
```
  @inproceedings{donteu2023sparse,
  title={Sparse Activations for Interpretable Disease Grading},
  author={Donteu, Kerol R Djoumessi and Ilanchezian, Indu and K{\"u}hlewein, Laura and Faber, Hanna and Baumgartner, Christian F and Bah, Bubacarr and Berens, Philipp and Koch, Lisa M},
  booktitle={Medical Imaging with Deep Learning},
  year={2023}
}
```
