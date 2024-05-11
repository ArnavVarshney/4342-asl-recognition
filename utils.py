import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchprofile import profile_macs
from sklearn.metrics import confusion_matrix
import seaborn as sns


dir_path = os.path.dirname(os.path.realpath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_accuracy(y_pred, y):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y).sum().item()
    total = y.size(0)
    return 100 * correct / total


def calculate_compression_rate(X, step_size):
    original_size = torch.numel(X) * X.element_size()
    quantized_size = torch.numel(X) * step_size.element_size()
    return original_size / quantized_size


def plot_curves(train_losses, train_accuracies, model_name=""):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, color='blue')
    plt.legend(['Original'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, color='blue')
    plt.legend(['Original'])
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, f'graphs/{model_name}'))


def get_file_size(filename) -> float:
    return os.path.getsize(filename) / 1e3


@torch.no_grad()
def sensitivity_scan(model, dataloader, scan_step=0.1, scan_start=0.4, scan_end=1.0, verbose=True):
    sparsities = np.arange(start=scan_start, stop=scan_end, step=scan_step)
    accuracies = []
    named_conv_weights = [(name, param) for (name, param) in model.named_parameters() if param.dim() > 1]
    for i_layer, (name, param) in enumerate(named_conv_weights):
        param_clone = param.detach().clone()
        accuracy = []
        for sparsity in sparsities:
            fine_grained_prune(param.detach(), sparsity=sparsity)
            acc = test(model, dataloader)
            if verbose:
                print(f'sparsity={sparsity:.4f}: accuracy={acc:.4f}% ', end='\n')
            # restore
            param.copy_(param_clone)
            accuracy.append(acc)
        accuracies.append(accuracy)
    return sparsities, accuracies


def plot_sensitivity_scan(model, sparsities, accuracies, dense_model_accuracy):
    lower_bound_accuracy = 100 - (100 - dense_model_accuracy) * 1.5
    fig, axes = plt.subplots(3, int(math.ceil(len(accuracies) / 3)), figsize=(15, 8))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            ax = axes[plot_index]
            print("accuracies: ", accuracies[plot_index], name)
            ax.plot(sparsities, accuracies[plot_index])
            ax.plot(sparsities, [lower_bound_accuracy] * len(sparsities))
            ax.set_xticks(np.arange(start=0, stop=1.0, step=0.1))
            ax.set_title(name)
            ax.set_xlabel('sparsity')
            ax.set_ylabel('top-1 accuracy')
            ax.legend(['accuracy after pruning',
                       f'{lower_bound_accuracy / dense_model_accuracy * 100:.0f}% of dense model accuracy'])
            ax.grid(axis='x')
            plot_index += 1
    fig.suptitle('Sensitivity Curves: Validation Accuracy vs. Pruning Sparsity')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.savefig(os.path.join(dir_path, 'graphs/sensitivity_scan.png'))
    plt.show()


def fine_grained_prune(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    sparsity = np.clip(sparsity, 0.0, 1.0)

    if sparsity == 1.0:
        tensor.zero_()
        return torch.zeros_like(tensor)
    elif sparsity == 0.0:
        return torch.ones_like(tensor)

    num_elements = tensor.numel()

    num_zeros = round(num_elements * sparsity)
    importance = torch.square(tensor)
    threshold = importance.view(-1).kthvalue(num_zeros).values + 1e-8
    mask = torch.gt(importance, threshold)

    tensor.mul_(mask)

    return mask.long()


def plot_num_parameters_distribution(model):
    num_parameters = dict()
    for name, param in model.named_parameters():
        if param.dim() > 1:
            num_parameters[name] = param.numel()
    print(num_parameters)
    plt.figure(figsize=(8, 6))
    plt.grid(axis='y')
    plt.bar(list(num_parameters.keys()), list(num_parameters.values()))
    plt.title('Parameter Distribution')
    plt.ylabel('Number of Parameters')
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, 'graphs/num_parameters_distribution.png'))
    plt.show()


def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def get_model_sparsity(model: nn.Module) -> float:
    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return float(num_nonzeros) / num_elements


def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    return get_num_parameters(model, count_nonzero_only) * data_width


@torch.no_grad()
def measure_latency(model, dummy_input, n_warmup=20, n_test=100):
    model.eval()
    for _ in range(n_warmup):
        _ = model(dummy_input)
    t1 = time.time()
    for _ in range(n_test):
        _ = model(dummy_input)
    t2 = time.time()
    return (t2 - t1) / n_test  # average latency


def get_model_macs(model, inputs) -> int:
    return profile_macs(model, inputs)


def train(model, train_loader, val_loader, optimizer, num_epochs):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        total_train_loss = 0
        total_train_correct = 0
        total_train_samples = 0

        for batch, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)

            model.train()
            optimizer.zero_grad()
            y_pred = model(X)
            loss = model.loss(y_pred, y.squeeze())
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            train_accuracy = calculate_accuracy(y_pred, y.squeeze())
            total_train_correct += train_accuracy * y.size(0)
            total_train_samples += y.size(0)

            print(f"Epoch [{epoch + 1}], Step [{batch + 1}], Train Loss: {loss.item():.4f}", end="\r")

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        train_accuracy = total_train_correct / total_train_samples
        train_accuracies.append(train_accuracy)

        total_val_loss = 0
        total_val_correct = 0
        total_val_samples = 0

        model.eval()
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)

                y_val_pred = model(X_val)
                val_loss = model.loss(y_val_pred, y_val.squeeze())
                total_val_loss += val_loss.item()

                val_accuracy = calculate_accuracy(y_val_pred, y_val.squeeze())
                total_val_correct += val_accuracy * y_val.size(0)
                total_val_samples += y_val.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        val_accuracy = total_val_correct / total_val_samples
        val_accuracies.append(val_accuracy)

        print(f"\nEpoch [{epoch + 1}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Epoch [{epoch + 1}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        if not os.path.exists(os.path.join(dir_path, 'temp')):
            os.makedirs(os.path.join(dir_path, 'temp'))

        with open(os.path.join(dir_path, f'temp/{model.__class__.__name__}.txt'), 'w') as f:
            f.write(f"{train_losses}\n")
            f.write(f"{train_accuracies}\n")
            f.write(f"{val_losses}\n")
            f.write(f"{val_accuracies}\n")

    plot_curves(train_losses, train_accuracies, model.__class__.__name__)


@torch.inference_mode()
def test(model, test_loader, int8=False) -> float:
    model.eval()
    correct, total = 0, 0

    for X, y in test_loader:
        X = X.to(device)
        y = y.to(device)

        if int8:
            X = (X * 255 - 128).clamp(-128, 127).to(torch.int8)

        predicted = model(X)
        predicted = predicted.argmax(dim=1)

        total += y.size(0)
        correct += (predicted == y.squeeze()).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    return 100 * correct / total

def get_project_root():
    return os.path.dirname(__file__)

def get_confusion_matrix(model, dataloader, device, class_labels):
    model.eval()
    model_name = model.__class__.__name__
    true_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())
    
    cm = confusion_matrix(true_labels, predicted_labels)

    print(true_labels)
    print(predicted_labels)

    plt.style.use('dark_background')
    plt.rcParams["font.family"] = 'Consolas'

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, "graphs", f'{model_name}_confusion_matrix.png'), transparent=True)
