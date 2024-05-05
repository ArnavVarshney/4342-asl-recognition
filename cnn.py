import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

import utils

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
        img = F.log_softmax(img, dim=1)
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
            X = X.to(device)
            y = y.to(device)

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
            X = X.to(device)
            y = y.to(device)

            predicted = model(X)
            _, predicted = torch.max(predicted, 1)
            total += y.size(0)
            correct += (predicted == y.squeeze()).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    return 100 * correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train", type=int, default=1)

    epochs = parser.parse_args().epochs
    batch_size = parser.parse_args().batch_size
    training = parser.parse_args().train

    args = parser.parse_args()

    train_loader, test_loader = dataset()
    model = CNN(1, num_classes)
    model.to(device)

    if os.path.exists("asl.pth") and not args.train:
        model.load_state_dict(torch.load("asl.pth"))
        model.eval()
        print(summary(model, (1, 28, 28)))

    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        print(summary(model, (1, 28, 28)))
        train(model, train_loader, optimizer, num_epochs=epochs)
        test(model, test_loader)
        torch.save(model.state_dict(), "asl.pth")
