import cv2
import numpy as np
import torch

from cnn import CNN

classnames = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z']

model = CNN(1, 26)
model.load_state_dict(torch.load('asl.pth'))

cap = cv2.VideoCapture(0)
width, height = 640, 480
cap.set(3, width)
cap.set(4, height)

top, bot, right, left = 0, 240, 400, 640

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    roi = frame[top:bot, right:left]
    roi = cv2.resize(roi, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    roi = np.reshape(roi, (1, 1, 28, 28))
    roi = torch.from_numpy(roi).type(torch.FloatTensor)

    outputs = model(roi)
    print(outputs)

    conf, predicted = torch.max(outputs.squeeze(), 0)

    cv2.putText(frame, classnames[predicted.item()], (32, 64), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 2)

    cv2.rectangle(frame, (left, top), (right, bot), (255, 0, 255), 2)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
