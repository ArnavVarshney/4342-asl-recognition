import argparse
import os

import torch
import cv2
import numpy as np

import tvm
from tvm.contrib import graph_runtime

from cnn import CNN

dirpath = os.path.dirname(os.path.realpath(__file__))

classnames = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 
              'h', 'i', 'j', 'k', 'l', 'm', 'n', 
              'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z']

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mnist-sign-language", choices=["mnist-sign-language", "rock-paper-scissors"])

args = parser.parse_args()

lib = tvm.runtime.load_module(f"{dirpath}/tvm_out/model.so")

with open(f"{dirpath}/tvm_out/graph.json", "r") as f:
    graph = f.read()
with open(f"{dirpath}/tvm_out/params.params", "rb") as f:
    params = bytearray(f.read())

ctx = tvm.cpu()
module = graph_runtime.create(graph, lib, ctx)
module.load_params(params)

print("Model loaded successfully")

cap = cv2.VideoCapture(0)
width, height = 640, 480
cap.set(3, width)
cap.set(4, height)

print("Camera initialized successfully")

top, bot, right, left = 0, 280, 360, 640
box_center = (right + left) // 2, (top + bot) // 2

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)

fps_start = cv2.getTickCount()
fps_count = 0
fps = 0

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        continue

    roi = frame[top:bot, right:left]
    roi = cv2.resize(roi, dsize=(28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = np.reshape(roi, (1, 1, 28, 28))
    roi = torch.from_numpy(roi).type(torch.FloatTensor)
    roi /= 255

    module.set_input("input.1", roi.numpy())
    module.run()

    output = module.get_output(0).asnumpy()
    predictions = np.argmax(output, axis=1)
    predicted_class = classnames[predictions[0]]

    roi = roi.squeeze().numpy()

    frame = cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    # frame = cv2.putText(frame, f"Confidence: {confidence:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255),
    #                     2)
    frame = cv2.putText(frame, f"Predicted: {predicted_class}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255),
                        2)
    frame = cv2.rectangle(frame, (left, top), (right, bot), (255, 0, 255), 2)
    frame = cv2.circle(frame, box_center, 3, (255, 0, 255), -1)

    cv2.imshow("Frame", frame)
    cv2.imshow("ROI", roi)

    fps_count += 1
    if cv2.getTickCount() - fps_start >= cv2.getTickFrequency():
        fps = fps_count / ((cv2.getTickCount() - fps_start) / cv2.getTickFrequency())
        fps_start = cv2.getTickCount()
        fps_count = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()