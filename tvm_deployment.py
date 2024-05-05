import tvm
from tvm import relay
from tvm.contrib import graph_runtime

import numpy as np
import os

import cnn

dirname = os.path.dirname(__file__)
lib = tvm.runtime.load_module(f"{dirname}/tvm_out/model.so")
with open(f"{dirname}/tvm_out/graph.json", "r") as f:
    graph = f.read()
with open(f"{dirname}/tvm_out/params.params", "rb") as f:
    params = bytearray(f.read())

ctx = tvm.cpu()
module = graph_runtime.create(graph, lib, ctx)

module.load_params(params)

_, test_loader = cnn.dataset()
correct, total = 0, 0

for X, y in test_loader:
    if X.size(0) != 128:
        continue
    module.set_input("input.1", X.numpy())
    module.run()

    output = module.get_output(0).asnumpy()
    predictions = np.argmax(output, axis=1)

    correct += np.sum(predictions == y.numpy().squeeze())
    total += y.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")