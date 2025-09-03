import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from PIL import Image
from io import StringIO

class DMRIRDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None, height=244, width=244):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

        self.height = height
        self.width = width

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        
        if img_path.endswith('.txt'):
            txt_file = os.path.join(img_path)
            img = self.load_txt_image(txt_file)

            label = 1 if label == "cancro" else 0 
            label = torch.tensor(label, dtype=torch.long)

        elif img_path.endswith('.jpg') or img_path.endswith('.jpeg'):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image file {img_path} not found or could not be read.")
        else:
            raise ValueError(f"Unsupported file format: {img_path}")

        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)

        img = np.array(img, dtype=np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        if self.transform:
            img = img * 255.0
            img = img.astype(np.uint8)

            img = self.transform(img)

            img = img / 255.0
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = img.astype(np.float32)
            img = np.clip(img, 0, 1)

        img = np.expand_dims(img, axis=0)
        img = torch.tensor(img, dtype=torch.float32)
        img = img.expand(3, -1, -1) 


        return img, label

    def load_txt_image(self, txt_file):
        with open(txt_file, 'r') as file:
            content = file.read()

        content = content.replace(',', '.')
        content_io = StringIO(content)  
        data = np.genfromtxt(content_io, delimiter=None)  
        data = np.array(data, dtype=np.float32)
        return data


