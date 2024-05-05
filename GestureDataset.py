import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

class GestureDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.classes = self.data['label']

        self.img = self.data.drop('label', axis=1)
        self.img = self.img / 255.0
        self.img = self.img.values.reshape(-1, 28, 28, 1)

        self.transform = T.Compose([T.ToPILImage(), T.RandomRotation(10),
                                    T.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5)),
                                    T.RandomResizedCrop(28, scale=(1.0, 2)), T.ToImage(),
                                    T.ToDtype(torch.float32, scale=True)])

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        label = self.classes[index]
        img = self.img[index]
        img = self.transform(img)

        label = torch.LongTensor([label])
        img = img.float()

        return img, label

def dataset(train_path, test_path, batch_size=128):
    train_dataset = GestureDataset(train_path)
    test_dataset = GestureDataset(test_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader