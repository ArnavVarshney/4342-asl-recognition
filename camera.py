import argparse
import os
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from cnn import CNN

dirpath = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mnist-sign-language",
                    choices=["mnist-sign-language", "rock-paper-scissors"])

args = parser.parse_args()

if args.model == "mnist-sign-language":
    classnames = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                  'u',
                  'v', 'w', 'x', 'y', 'z']
    model = CNN(1, 26)
elif args.model == "rock-paper-scissors":
    classnames = ['Rock', 'Paper', 'Scissors', 'Empty']
    model = CNN(1, 4)

model.load_state_dict(torch.load(f'{dirpath}/weights/{args.model}/model.pth'))

print("Model loaded successfully")


def get_color_bounds(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[width // 2, height // 2]

    color = np.uint8([[hsv]])
    color = cv2.cvtColor(color, cv2.COLOR_HSV2RGB)
    plt.imshow(color)
    plt.show()

    lower_bound = np.array([hsv[0] - 10, hsv[1] - 25, hsv[2] - 50])
    upper_bound = np.array([hsv[0] + 10, hsv[1] + 25, hsv[2] + 50])

    return lower_bound, upper_bound


def get_largest_contour(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)

    if len(contours) == 0:
        return mask

    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)

    return mask


def get_bounding_box(image, padding=0):
    x, y, w, h = cv2.boundingRect(image)
    square_size = max(w, h)

    centroid = (x + w // 2, y + h // 2)

    x1 = max(centroid[0] - square_size // 2 - padding, 0)
    y1 = max(centroid[1] - square_size // 2 - padding, 0)
    x2 = min(centroid[0] + square_size // 2 + padding, image.shape[1])
    y2 = min(centroid[1] + square_size // 2 + padding, image.shape[0])

    return x1, y1, x2, y2


lower_bound, upper_bound = None, None

cap = cv2.VideoCapture(0)
width, height = 640, 480
cap.set(3, width)
cap.set(4, height)

cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    cv2.circle(frame, (width // 2, height // 2), 5, (0, 255, 0), -1)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        lower_bound, upper_bound = get_color_bounds(frame)
        break

fps_start = cv2.getTickCount()
fps_count = 0
fps = 0

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        continue

    mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lower_bound, upper_bound)
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    largest_contour = get_largest_contour(mask)

    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)

    result = cv2.bitwise_and(frame, frame, mask=mask)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    result[result == 0] = 255

    x1, y1, x2, y2 = get_bounding_box(largest_contour, 48)

    cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 255), 2)

    roi = deepcopy(result[y1:y2, x1:x2])
    roi = cv2.resize(roi, dsize=(28, 28), interpolation=cv2.INTER_AREA)

    roi = np.reshape(roi, (1, 1, 28, 28))
    roi = torch.from_numpy(roi).type(torch.FloatTensor)
    roi /= 255

    outputs = model(roi)
    conf, predicted = torch.max(outputs.squeeze(), 0)
    conf = torch.nn.functional.softmax(outputs, dim=1)[0][predicted] * 100
    predicted_class = classnames[predicted.item()]
    confidence = conf.item()

    roi = roi.squeeze().numpy()

    frame = cv2.putText(result, f"FPS: {fps}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),
                        2)
    frame = cv2.putText(result, f"Confidence: {confidence:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),
                        2)
    frame = cv2.putText(result, f"Predicted: {predicted_class}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),
                        2)

    cv2.imshow("Result", result)
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
