import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import GestureDataset
from utils import train, test

batch_size = 128
num_classes = 26
epochs = 10
dirname = os.path.dirname(__file__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)

    epochs = parser.parse_args().epochs
    batch_size = parser.parse_args().batch_size
    training = parser.parse_args().train
    lr = parser.parse_args().lr

    args = parser.parse_args()

    train_loader, test_loader = GestureDataset.dataset(os.path.join(dirname, 'mnist-sign-language/train/sign_mnist_train.csv'), 
                                                       os.path.join(dirname, 'mnist-sign-language/test/sign_mnist_test.csv'), 
                                                       args.batch_size)
    
    if not os.path.exists(f"{dirname}/weights"):
        os.makedirs(f"{dirname}/weights")

    model = CNN()
    model.to(device)

    if os.path.exists(f"{dirname}/weights/asl.pth") and args.train:
        model.load_state_dict(torch.load(f"{dirname}/weights/asl.pth"))
        model.eval()
        print(summary(model, (1, 28, 28)))

    else:
        print(summary(model, (1, 28, 28)))
        train(model, train_loader, torch.optim.Adam(model.parameters(), lr=1e-3), num_epochs=epochs)
        test(model, test_loader)
        torch.save(model.state_dict(), f"{dirname}/weights/asl.pth")