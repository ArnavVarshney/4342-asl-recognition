#DONT RUN THIS BY ITSELF
#---------------------------------



# import ncnn
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
import torch.onnx
import torchvision
import os
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
import pandas as pd


batch_size = 128
num_classes = 26
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dir = "../ncnn_out"
if not os.path.exists(dir):
        os.makedirs(dir)


class CNN(nn.Module):
    def __init__(self, in_channels=1, out_features=num_classes):
        super(CNN, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, out_features)
        )

    def forward(self, img):
        img = self.conv1(img)
        img = self.conv2(img)
        img = self.conv3(img)
        img = self.fc(img)
        return img

    def loss(self, x, label):
        loss = self.criterion(x, label)
        return loss
    

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

train_loader, test_loader = dataset('../../mnist-sign-language/mnist_sign_language_train.csv', 
                                                     '../../mnist-sign-language/mnist_sign_language_test.csv', 
                                                    128)


#Converting to NCNN type
#Source: https://github.com/Tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx
weights = torch.load("../../weights/mnist-sign-language/model.pth")
model = CNN()
model.load_state_dict(weights)
model.eval()

torch_out = torch.onnx._export(model, torch.rand(128, 1, 28, 28), dir+"/model.onnx", export_params=True) # Export the model
os.system(f"python3 -m onnxsim {dir+"/model.onnx"} {dir+"/model-sim.onnx"}")
os.system(f"onnx2ncnn {dir+"/model-sim.onnx"} {dir+"/model.param"} {dir+"/model.bin"}")