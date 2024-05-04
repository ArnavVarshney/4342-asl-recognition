import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

import utils

batch_size = 128
num_classes = 26
epochs = 20
dirname = os.path.dirname(__file__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels * self.expansion)
        )

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=1):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.criterion = nn.CrossEntropyLoss()

        self.initial_layer = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.initial_layer(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def loss(self, x, label):
        loss = self.criterion(x, label)
        return loss

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


class GestureDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.classes = self.data['label']

        self.img = self.data.drop('label', axis=1)
        self.img = self.img / 255.0
        self.img = self.img.values.reshape(-1, 28, 28, 1)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5)),
            T.RandomResizedCrop(28, scale=(1.0, 2)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True)
        ])

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        label = self.classes[index]
        img = self.img[index]
        img = self.transform(img)

        label = torch.LongTensor([label])
        img = img.float()

        return img, label


def dataset():
    train_dataset = GestureDataset(os.path.join(dirname, 'mnist-sign-language/train/sign_mnist_train.csv'))
    test_dataset = GestureDataset(os.path.join(dirname, 'mnist-sign-language/test/sign_mnist_test.csv'))

    # train_dataset = GestureDataset(os.path.join(dirname, 'train.csv'))
    # test_dataset = GestureDataset(os.path.join(dirname, 'test.csv'))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def train(model, train_loader, optimizer, num_epochs):
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = model.loss(y_pred, y.squeeze())
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_accuracy = utils.calculate_accuracy(y_pred, y.squeeze())
            total_correct += train_accuracy * y.size(0)
            total_samples += y.size(0)

            print(
                f"Epoch [{epoch + 1}], Step [{batch + 1}], Loss: {loss.item():.4f}",
                end="\r",
            )

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        train_accuracy = total_correct / total_samples
        train_accuracies.append(train_accuracy)

        print(
            f"\nEpoch [{epoch + 1}], Average Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%"
        )

    utils.plot_curves(train_losses, train_accuracies)


def test(model, test_loader):
    model.eval()
    correct, total = 0, 0

    with torch.inference_mode():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            predicted = model(X)
            _, predicted = torch.max(predicted, 1)
            total += y.size(0)
            correct += (predicted == y.squeeze()).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    return 100 * correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--train", type=bool, default=True)
    args = parser.parse_args()

    train_loader, test_loader = dataset()
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, 1)
    model.to(device)

    if os.path.exists("asl.pth") and not args.train:
        model.load_state_dict(torch.load("asl.pth"))
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        train(model, train_loader, optimizer, num_epochs=args.epochs)
        test(model, test_loader)
        torch.save(model.state_dict(), "asl.pth")
