import os

import cv2
import numpy as np
import torch
import argparse

from cnn import CNN

dirpath = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mnist-sign-language", choices=["mnist-sign-language", "rock-paper-scissors"])

args = parser.parse_args()

if args.model == "mnist-sign-language":
    classnames = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y']
    model = CNN(1, 24)
elif args.model == "rock-paper-scissors":
    classnames = ['Rock', 'Paper', 'Scissors', 'Empty']
    model = CNN(1, 4)

model.load_state_dict(torch.load(f'{dirpath}/weights/{args.model}/model.pth'))

print("Model loaded successfully")

cap = cv2.VideoCapture(0)
width, height = 640, 480
cap.set(3, width)
cap.set(4, height)

print("Camera initialized successfully")

top, bot, right, left = 0, 240, 400, 640

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
    roi = cv2.resize(roi, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    roi = np.reshape(roi, (1, 1, 28, 28))
    roi = torch.from_numpy(roi).type(torch.FloatTensor)
    roi /= 255

    outputs = model(roi)
    conf, predicted = torch.max(outputs.squeeze(), 0)
    conf = torch.nn.functional.softmax(outputs, dim=1)[0][predicted] * 100
    predicted_class = classnames[predicted.item()]
    confidence = conf.item()

    roi = roi.squeeze().numpy()

    frame = cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    frame = cv2.putText(frame, f"Confidence: {confidence:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255),
                        2)
    frame = cv2.putText(frame, f"Predicted: {predicted_class}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255),
                        2)
    frame = cv2.rectangle(frame, (left, top), (right, bot), (255, 0, 255), 2)

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