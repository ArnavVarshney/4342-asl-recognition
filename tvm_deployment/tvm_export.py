import os

import onnx
import torch
import tvm
from tvm import relay

import cnn
from utils import get_project_root

root = get_project_root()
dirname = root + "/tvm_deployment"
device = torch.device("cpu")

if not os.path.exists(f"{dirname}/tvm_out"):
    os.makedirs(f"{dirname}/tvm_out")

model = cnn.CNN()
model.to(device)
model.load_state_dict(torch.load(f"{root}/weights/mnist-sign-language/model.pth"))

torch.onnx.export(model, torch.randn(128, 1, 28, 28), f"{dirname}/tvm_out/model.onnx")
onnx_model = onnx.load(f"{dirname}/tvm_out/model.onnx")

input_name = onnx_model.graph.input[0].name
print(input_name)
input_shape = tuple(d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim)
shape_dict = {input_name: input_shape}
print(input_shape)

mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

with tvm.transform.PassContext(opt_level=3):
    # graph_module = relay.build(mod, target="llvm", params=params)
    graph_module = relay.build(mod, target=tvm.target.arm_cpu("rpi4b64"), params=params)

graph = graph_module.get_graph_json()
lib = graph_module.get_lib()
params = graph_module.get_params()

lib.export_library(f"{dirname}/tvm_out/model.so")

with open(f"{dirname}/tvm_out/graph.json", "w") as f:
    f.write(graph)

with open(f"{dirname}/tvm_out/params.params", "wb") as f:
    f.write(relay.save_param_dict(params))
