import cv2
import torch
import numpy as np
import cnn as cnn

classnames = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

model = cnn.CNN()
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
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.GaussianBlur(roi, (3, 3), 0)

    roi = np.asarray(roi, dtype=np.float32)
    roi = cv2.resize(roi, (28, 28))
    roi = roi / 255.0

    predict = torch.tensor(roi).unsqueeze(0).unsqueeze(0)

    outputs = model(predict)
    _, predicted = torch.max(outputs.squeeze(), 0)
    cv2.putText(frame, classnames[predicted.item()], (32, 64), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 2)
    print(classnames[predicted.item()])

    cv2.rectangle(frame, (left, top), (right, bot), (255, 0, 255), 2)

    cv2.imshow("frame", frame)
    cv2.imshow("ROI", roi)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()