import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import Dataset
from torchsummary import summary
from torchvision.transforms import v2 as T

import utils
import pandas as pd
import cv2 as cv2
import os
import argparse
import LSQ
import cnn

batch_size = 128
num_classes = 26
epochs = 10
dirname = os.path.dirname(__file__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN_Quantized(nn.Module):
    def __init__(self, model, bits=8):
        super(CNN_Quantized, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

        self.conv1 = nn.Sequential(
            LSQ.Conv2d(
                model.conv1[0],
                LSQ.Quantizer(8, True),
                LSQ.Quantizer(8, False),
            ),
            LSQ.BatchNorm2d(
                model.conv1[1],
                LSQ.Quantizer(8, True),
                LSQ.Quantizer(8, False),
            ),
            nn.LeakyReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            LSQ.Conv2d(
                model.conv2[0],
                LSQ.Quantizer(8, True),
                LSQ.Quantizer(8, False),
            ),
            LSQ.BatchNorm2d(
                model.conv2[1],
                LSQ.Quantizer(8, True),
                LSQ.Quantizer(8, False),
            ),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.conv3 = nn.Sequential(
            LSQ.Conv2d(
                model.conv3[0],
                LSQ.Quantizer(8, True),
                LSQ.Quantizer(8, False),
            ),
            LSQ.BatchNorm2d(
                model.conv3[1],
                LSQ.Quantizer(8, True),
                LSQ.Quantizer(8, False),
            ),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            LSQ.Linear(
                model.fc[1],
                LSQ.Quantizer(8, True),
                LSQ.Quantizer(8, False),
            ),
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

    @torch.no_grad()
    def change_datatype(self):
        params = self.state_dict()
        params["conv1.0.weight"] = torch.round(params["conv1.0.weight"] / self.conv1[0].weight_quantizer.step_size).to(torch.int8)
        params["conv1.1.weight"] = torch.round(params["conv1.1.weight"] / self.conv1[1].weight_quantizer.step_size).to(torch.int8)
        params["conv2.0.weight"] = torch.round(params["conv2.0.weight"] / self.conv2[0].weight_quantizer.step_size).to(torch.int8)
        params["conv2.1.weight"] = torch.round(params["conv2.1.weight"] / self.conv2[1].weight_quantizer.step_size).to(torch.int8)
        params["conv3.0.weight"] = torch.round(params["conv3.0.weight"] / self.conv3[0].weight_quantizer.step_size).to(torch.int8)
        params["conv3.1.weight"] = torch.round(params["conv3.1.weight"] / self.conv3[1].weight_quantizer.step_size).to(torch.int8)
        params["fc.1.weight"] = torch.round(params["fc.1.weight"] / self.fc[1].weight_quantizer.step_size).to(torch.int8)
        
        self.load_state_dict(params)
        torch.save(params, "cnn_quant_weights.pth")


class GestureDataset(Dataset) :
    def __init__(self, csv_file) :
        self.data = pd.read_csv(csv_file)
        self.classes = self.data['label']

        self.img = self.data.drop('label', axis=1)
        self.img = self.img / 255.0
        self.img = self.img.values.reshape(-1, 28, 28, 1)
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=(0.5,1.5), contrast=(0.5,1.5), saturation=(0.5,1.5)),
            T.RandomResizedCrop(28, scale=(1.0, 2)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True)
        ])

    def __len__(self) :
        return len(self.img)
    
    def __getitem__(self, index) :
        label = self.classes[index]
        img = self.img[index]
        img = self.transform(img)
        
        label = torch.LongTensor([label])
        img = img.float()
        
        return img, label
    
def dataset():
    train_dataset = GestureDataset(os.path.join(dirname, 'mnist-sign-language/train/sign_mnist_train.csv'))
    test_dataset = GestureDataset(os.path.join(dirname, 'mnist-sign-language/test/sign_mnist_test.csv'))
    
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

    return train_loader,test_loader

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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

def test(model, test_loader, extra_preprocess=False) -> float:
    model.eval()
    correct, total = 0, 0

    for X, y in test_loader:
        X = X.to(device)
        y = y.to(device)

        if extra_preprocess == True:
            X = to_int_8(X)

        predicted = model(X)
        predicted = predicted.argmax(dim=1)

        total += y.size(0)
        correct += (predicted == y.squeeze()).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    return 100 * correct / total

def to_int_8(x):
    return (x * 255 - 128).clamp(-128, 127).to(torch.int8)

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

    model = cnn.CNN(1, num_classes)
    model.to(device)
    model.load_state_dict(torch.load("asl.pth"))

    test(model, test_loader)
    print(utils.get_file_size("asl.pth"))

    quant_model = CNN_Quantized(model, 8)
    quant_model.to(device)

    if os.path.exists("asl.pth"):
        if os.path.exists("asl_quant.pth"):
            quant_model.load_state_dict(torch.load("asl_quant.pth"))
        else:
            train_optimizer = torch.optim.Adam(quant_model.parameters(), lr=1e-3)
            train(quant_model, train_loader, train_optimizer, epochs)
            print(summary(quant_model, (1, 28, 28)))
            torch.save(quant_model.state_dict(), "asl_quant.pth")
        
        print(utils.get_file_size("asl_quant.pth"))
        test(quant_model, test_loader)

        quant_model.change_datatype()
        print(utils.get_file_size("cnn_quant_weights.pth"))
        test(quant_model, test_loader, extra_preprocess=True)

    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        print(summary(model, (1, 28, 28)))
        train(model, train_loader, optimizer, num_epochs=epochs)
        test(model, test_loader, extra_preprocess=True)
        torch.save(model.state_dict(), "asl.pth")