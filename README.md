MQFusionNet: An Adaptive Multi-Query Feature Fusion Network for Multi-Label Chest X-Ray Classification
========================================================================================================

Official PyTorch implementation of the paper:
"MQFusionNet: An Adaptive Multi-Query Feature Fusion Network for Multi-Label Classification of Chest X-Ray Images"

1. Project Overview
-------------------
This repository contains the source code for MQFusionNet. The model integrates three heterogeneous backbones (VGG16, ResNet50, DenseNet121) using a Cross-Net Query Attention Module to effectively handle label co-occurrence and class imbalance in chest X-ray analysis.

2. Prerequisites & Installation
-------------------------------
(1) Create a virtual environment:
    conda create -n mqfusion python=3.9
    conda activate mqfusion

(2) Install dependencies:
    pip install -r requirements.txt

3. Data Preparation
-------------------
(1) Download the Dataset:
    - Download 'Data_Entry_2017.csv' and the image folders from the official ChestX-ray14 website (NIH Clinical Center).
    - Unzip all images into a single folder, e.g., `/datasets/ChestX-ray14/images/`.

(2) Generate Data Splits (Crucial for Reproducibility):
    Run the split script to generate train/val/test CSVs with patient-level non-overlapping constraint (Seed=42).

    python split_data.py \
        --input_csv /datasets/ChestX-ray14/Data_Entry_2017.csv \
        --output_dir ./data_splits

    This will create 'train_labels.csv', 'val_labels.csv', and 'test_labels.csv' in the './data_splits' folder.

4. Training
-----------
To reproduce the training configuration (Section III-B):
- Epochs: 120
- Batch Size: 32
- Learning Rate: 1e-4
- Lambda (Regularization): 0.1
- Seed: 3407

Run the following command:
python train.py \
    --data_dir /datasets/ChestX-ray14/images \
    --train_csv ./data_splits/train_labels.csv \
    --val_csv ./data_splits/val_labels.csv \
    --save_dir ./checkpoints

5. Evaluation
-------------
To test the trained model on the held-out test set (20%) and generate AUC/MAE metrics:

python test.py \
    --data_dir /datasets/ChestX-ray14/images \
    --test_csv ./data_splits/test_labels.csv \
    --weights ./checkpoints/best_model.pth

6. Citation
-----------
If you use this code in your research, please cite our paper.