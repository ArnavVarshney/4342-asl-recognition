import os

import onnx
import torch

import cnn
from utils import get_project_root

root = get_project_root()
dirname = root + "/mnn_deployment"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mnn_path = "~/Dev/MNN/build"

if not os.path.exists(f"{dirname}/mnn_out"):
    os.makedirs(f"{dirname}/mnn_out")

model = cnn.CNN()
model.to(device)
model.load_state_dict(torch.load(f"{root}/weights/asl.pth"))

print(f"{dirname}/mnn_out")
torch.onnx.export(model, torch.randn(128, 1, 28, 28), f"{dirname}/mnn_out/model.onnx")
onnx_model = onnx.load(f"{dirname}/mnn_out/model.onnx")

os.system(
    f"{mnn_path}/MNNConvert -f ONNX --modelFile {dirname}/mnn_out/model.onnx --MNNModel {dirname}/mnn_out/model.mnn")
