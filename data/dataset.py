import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pandas as pd
import numpy as np

class ChestXRayDataset(Dataset):
    """
    Standard loader for ChestX-ray14 and PadChest datasets.
    Handles multi-label binary vectors and image preprocessing.
    """
    def __init__(self, root_dir, csv_file, transform=None):
        """
        Args:
            root_dir (str): Path to image directory.
            csv_file (str): Path to the csv file with annotations.
            transform (callable, optional): Augmentations.
        """
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
        # 14 Pathological classes as defined in the manuscript
        self.classes = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image Index']
        img_path = os.path.join(self.root_dir, img_name)
        
        # Load Image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load image {img_path}. Using black image.")
            image = Image.new('RGB', (224, 224))

        # Get multi-hot labels
        labels = np.zeros(len(self.classes), dtype=np.float32)
        for i, cls_name in enumerate(self.classes):
            if cls_name in row and row[cls_name] == 1:
                labels[i] = 1.0

        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(labels)

def get_transforms(phase='train'):
    # Preprocessing pipeline as described in Section III-B
    # Mean and Std for normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)), # Mild augmentation to prevent overfitting
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])