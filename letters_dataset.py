import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LettersDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}

        for idx, folder in enumerate(sorted(os.listdir(root_dir))):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                self.class_to_idx[folder] = idx
                for file in os.listdir(folder_path):
                    if file.endswith(('.jpg','.png')):
                        self.images.append(os.path.join(folder_path, file))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('L')  # grayscale

        if self.transform:
            image = self.transform(image)

        return image, label
