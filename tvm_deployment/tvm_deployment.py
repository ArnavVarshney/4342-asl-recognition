import os
import time

import numpy as np
import tvm
from tvm.contrib import graph_runtime

import GestureDataset

dirname = os.path.dirname(__file__)
lib = tvm.runtime.load_module(f"{dirname}/tvm_out/model.so")

with open(f"{dirname}/tvm_out/graph.json", "r") as f:
    graph = f.read()
with open(f"{dirname}/tvm_out/params.params", "rb") as f:
    params = bytearray(f.read())

ctx = tvm.cpu()
module = graph_runtime.create(graph, lib, ctx)

module.load_params(params)

_, test_loader = GestureDataset.dataset(
    os.path.join(dirname, f'../mnist-sign-language/mnist_sign_language_test.csv'),
    os.path.join(dirname, f'../mnist-sign-language/mnist_sign_language_test.csv'),
    1
)

correct, total = 0, 0
dummy = next(iter(test_loader))[0].numpy()

for _ in range(20):
    module.set_input("input.1", dummy)
    module.run()

time_start = time.time()
for X, y in test_loader:
    module.set_input("input.1", X.numpy())
    module.run()

    output = module.get_output(0).asnumpy()
    predictions = np.argmax(output, axis=1)

    correct += np.sum(predictions == y.numpy().squeeze())
    total += y.size(0)
time_end = time.time()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
print(f"Latency: {(time_end - time_start) / len(test_loader) * 1000:.3f} ms per image")