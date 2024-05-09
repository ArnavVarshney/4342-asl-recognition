import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import LSQ
import cnn
from utils import train, test, fine_grained_prune, get_file_size
import GestureDataset

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

        torch.save(new_params, f"{dirname}/weights/cnn_pruned_weights.pth")

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

        torch.save(params, f"{dirname}/weights/cnn_quant_weights.pth")
        torch.save(steps_dict, f"{dirname}/weights/cnn_quant_steps.pth")
    
    @torch.no_grad()
    def load_quantized_params(self):
        state_dict = torch.load(f"{dirname}/weights/cnn_quant_weights.pth")
        steps = torch.load(f"{dirname}/weights/cnn_quant_steps.pth")

        for key in state_dict.keys():
            if "weight" in key:
                state_dict[key] = state_dict[key] * steps[key]
        self.load_state_dict(state_dict)

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
    
    # Original Model
    model = cnn.CNN()
    model.to(device)

    start_time = time.time()
    if os.path.exists(f"{dirname}/weights/asl.pth"):
        model.load_state_dict(torch.load(f"{dirname}/weights/asl.pth"))
        model.eval()
        print(summary(model, (1, 28, 28)))
    else:
        print(summary(model, (1, 28, 28)))
        train(model, train_loader, torch.optim.Adam(model.parameters(), lr=1e-3), args.epochs)
        torch.save(model.state_dict(), f"{dirname}/weights/asl.pth")
        print("Training time: ", time.time() - start_time)

    test(model, test_loader)

    # Quantized Model
    quant_model = CNN_Quantized(model)
    quant_model.to(device)

    start_time = time.time()
    if os.path.exists(f"{dirname}/weights/asl_quant.pth"):
        quant_model.load_state_dict(torch.load(f"{dirname}/weights/asl_quant.pth"))
        quant_model.eval()
        print(summary(quant_model, (1, 28, 28)))
    else:
        print(summary(quant_model, (1, 28, 28)))
        train(quant_model, train_loader, torch.optim.Adam(quant_model.parameters(), lr=args.lr), args.epochs)
        torch.save(quant_model.state_dict(), f"{dirname}/weights/asl_quant.pth")
        print("Training time: ", time.time() - start_time)

    quant_model.change_datatype()
    quant_model.load_quantized_params()
    print("Accuracy after quantization and change datatype: ")
    test(quant_model, test_loader)

    size_quantized = get_file_size(f"{dirname}/weights/cnn_quant_weights.pth") \
                     + get_file_size(f"{dirname}/weights/cnn_quant_steps.pth")

    size_original = get_file_size(f"{dirname}/weights/asl.pth")

    print(f"Before: {size_original} bytes, Quant: {size_quantized} bytes, Ratio: {size_quantized / size_original * 100:.2f}%")

    sparsity_dict = {"conv1.0.weight": 0.1, "conv2.0.weight": 0.50, "conv3.0.weight": 0.6, "fc.1.weight": 0.70, }
    quant_model.magnitude_prune(sparsity_dict)

    print("Accuracy after magnitude pruning: ")
    test(quant_model, test_loader)
