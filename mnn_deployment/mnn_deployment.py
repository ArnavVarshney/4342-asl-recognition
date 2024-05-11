import os

import MNN.nn as nn
import MNN.expr as expr
import MNN.numpy as np

import GestureDataset

dirname = os.path.dirname(__file__)

config = {}
config['precision'] = 'high'
config['backend'] = 0
config['numThread'] = 1

rt = nn.create_runtime_manager((config, ))
net = nn.load_module_from_file(f"{dirname}/mnn_out/model.mnn", [], [], runtime_manager=rt)

_, test_loader = GestureDataset.dataset(
    os.path.join(dirname, f'../mnist-sign-language/mnist_sign_language_test.csv'),
    os.path.join(dirname, f'../mnist-sign-language/mnist_sign_language_test.csv'),
    1
)

correct, total = 0, 0

for X, y in test_loader:
    input_data = expr.convert(X.numpy(), expr.NC4HW4)
    output = net.forward(input_data)
    output = expr.convert(output, expr.NHWC)
    predictions = np.argmax(output, axis=1)

    correct += np.sum(predictions == y.numpy().squeeze())
    total += y.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")
