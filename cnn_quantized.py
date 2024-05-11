import argparse
import os
import time

import torch
import torch.nn as nn
from torchsummary import summary

import GestureDataset
import LSQ
import cnn
from utils import train, test, fine_grained_prune, get_file_size

batch_size = 128
num_classes = 26
lr = 0.001
epochs = 10
dirname = os.path.dirname(__file__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN_Quantized(nn.Module):
    def __init__(self, model, bits=8):
        super(CNN_Quantized, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

        self.conv1 = nn.Sequential(LSQ.Conv2d(model.conv1[0], LSQ.Quantizer(8, True), LSQ.Quantizer(8, False)),
                                   LSQ.BatchNorm2d(model.conv1[1], LSQ.Quantizer(8, True), LSQ.Quantizer(8, False)),
                                   nn.LeakyReLU(inplace=True))

        self.conv2 = nn.Sequential(LSQ.Conv2d(model.conv2[0], LSQ.Quantizer(8, True), LSQ.Quantizer(8, False)),
                                   LSQ.BatchNorm2d(model.conv2[1], LSQ.Quantizer(8, True), LSQ.Quantizer(8, False)),
                                   nn.LeakyReLU(inplace=True), nn.MaxPool2d(2), )

        self.conv3 = nn.Sequential(LSQ.Conv2d(model.conv3[0], LSQ.Quantizer(8, True), LSQ.Quantizer(8, False)),
                                   LSQ.BatchNorm2d(model.conv3[1], LSQ.Quantizer(8, True), LSQ.Quantizer(8, False)),
                                   nn.LeakyReLU(inplace=True), nn.MaxPool2d(2), )

        self.fc = nn.Sequential(nn.Flatten(),
                                LSQ.Linear(model.fc[1], LSQ.Quantizer(8, True), LSQ.Quantizer(8, False)), )

    def forward(self, img):
        img = self.conv1(img)
        img = self.conv2(img)
        img = self.conv3(img)
        img = self.fc(img)
        return img

    def loss(self, x, label):
        loss = self.criterion(x, label)
        return loss

    @torch.no_grad()
    def magnitude_prune(self, sparsity_dict):
        new_params = self.state_dict()
        masks = {}
        for name, param in self.named_parameters():
            if param.dim() > 1 and "weight" in name:
                masks[name] = fine_grained_prune(param, sparsity_dict[name])

        for name, param in self.named_parameters():
            if name in masks:
                new_params[name] = param * masks[name]
            if "bias" in name:
                new_params[name] = param
        self.load_state_dict(new_params)

        torch.save(new_params, f"{dirname}/weights/{dataset}/cnn_pruned_weights.pth")

    @torch.no_grad()
    def change_datatype(self):
        params = self.state_dict()

        steps_dict = {"conv1.0.weight": self.conv1[0].weight_quantizer.step_size,
                      "conv1.1.weight": self.conv1[1].weight_quantizer.step_size,
                      "conv2.0.weight": self.conv2[0].weight_quantizer.step_size,
                      "conv2.1.weight": self.conv2[1].weight_quantizer.step_size,
                      "conv3.0.weight": self.conv3[0].weight_quantizer.step_size,
                      "conv3.1.weight": self.conv3[1].weight_quantizer.step_size,
                      "fc.1.weight": self.fc[1].weight_quantizer.step_size, }

        for key in steps_dict.keys():
            if "weight" in key:
                params[key] = torch.round(params[key] / steps_dict[key]).to(torch.int8)

        torch.save(params, f"{dirname}/weights/{dataset}/cnn_quant_weights.pth")
        torch.save(steps_dict, f"{dirname}/weights/{dataset}/cnn_quant_steps.pth")

    @torch.no_grad()
    def load_quantized_params(self):
        state_dict = torch.load(f"{dirname}/weights/{dataset}/cnn_quant_weights.pth")
        steps = torch.load(f"{dirname}/weights/{dataset}/cnn_quant_steps.pth")

        for key in state_dict.keys():
            if "weight" in key:
                state_dict[key] = state_dict[key] * steps[key]
        self.load_state_dict(state_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset", type=str, default="mnist-sign-language", choices=["mnist-sign-language", "rock-paper-scissors"])

    epochs = parser.parse_args().epochs
    batch_size = parser.parse_args().batch_size
    training = parser.parse_args().train
    lr = parser.parse_args().lr
    dataset = parser.parse_args().dataset

    train_loader, test_loader = GestureDataset.dataset(
        os.path.join(dirname, f'{dataset}/{dataset.replace("-", "_")}_train.csv'),
        os.path.join(dirname, f'{dataset}/{dataset.replace("-", "_")}_test.csv'), 
        batch_size)

    if not os.path.exists(f"{dirname}/weights/{dataset}"):
        os.makedirs(f"{dirname}/weights/{dataset}")

    # Original Model
    if dataset == "mnist-sign-language":
        model = cnn.CNN(1, 26).to(device)
    elif dataset == "rock-paper-scissors":
        model = cnn.CNN(1, 4).to(device)

    start_time = time.time()
    if os.path.exists(f"{dirname}/weights/{dataset}/model.pth"):
        model.load_state_dict(torch.load(f"{dirname}/weights/{dataset}/model.pth"))
        model.eval()
        print(summary(model, (1, 28, 28)))
    else:
        print(summary(model, (1, 28, 28)))
        train(model, train_loader, test_loader, torch.optim.Adam(model.parameters(), lr=1e-3), epochs)
        torch.save(model.state_dict(), f"{dirname}/weights/{dataset}/model.pth")
        print(f"Training time: {time.time() - start_time:.2f}s")

    test(model, test_loader)

    # Quantized Model
    quant_model = CNN_Quantized(model).to(device)

    start_time = time.time()
    if os.path.exists(f"{dirname}/weights/{dataset}/model_quant.pth") and not training:
        quant_model.load_state_dict(torch.load(f"{dirname}/weights/{dataset}/model_quant.pth"))
        quant_model.eval()
        print(summary(quant_model, (1, 28, 28)))
    else:
        print(summary(quant_model, (1, 28, 28)))
        train(quant_model, train_loader, test_loader, torch.optim.Adam(quant_model.parameters(), lr=lr), epochs)
        torch.save(quant_model.state_dict(), f"{dirname}/weights/{dataset}/model_quant.pth")
        print(f"Training time: {time.time() - start_time:.2f}s")

    quant_model.change_datatype()
    quant_model.load_quantized_params()
    print("Accuracy after quantization and change datatype: ")
    test(quant_model, test_loader)

    size_quantized = get_file_size(f"{dirname}/weights/{dataset}/cnn_quant_weights.pth") + get_file_size(
        f"{dirname}/weights/{dataset}/cnn_quant_steps.pth")

    size_original = get_file_size(f"{dirname}/weights/{dataset}/model.pth")

    print(
        f"Before: {size_original} bytes, Quant: {size_quantized} bytes, Ratio: {size_quantized / size_original * 100:.2f}%")

    sparsity_dict = {"conv1.0.weight": 0.1, "conv2.0.weight": 0.50, "conv3.0.weight": 0.6, "fc.1.weight": 0.70, }
    quant_model.magnitude_prune(sparsity_dict)

    print("Accuracy after magnitude pruning: ")
    test(quant_model, test_loader)