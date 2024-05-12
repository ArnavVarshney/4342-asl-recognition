import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from copy import deepcopy
from sklearn.cluster import KMeans


def get_color_bounds(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[
        width // 2 - 5 : width // 2 + 5, 
        height // 2 - 5 : height // 2 + 5,
        :
    ].mean(axis=(0, 1))

    color = np.uint8([[hsv]])
    color = cv2.cvtColor(color, cv2.COLOR_HSV2RGB)
    plt.imshow(color)
    plt.show()

    lower_bound = np.array([hsv[0] - 10, hsv[1] - 25, hsv[2] - 50])
    upper_bound = np.array([hsv[0] + 10, hsv[1] + 25, hsv[2] + 50])

    return lower_bound, upper_bound

lower_bound, upper_bound = None, None

cap = cv2.VideoCapture(0)
width, height = 640, 480
cap.set(3, width)
cap.set(4, height)

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()

    cv2.circle(frame, (width // 2, height // 2), 5, (0, 255, 0), -1)

    if not ret or frame is None:
        continue

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        lower_bound, upper_bound = get_color_bounds(frame)
        break

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        continue

    img = frame

    mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lower_bound, upper_bound)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(mask)
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    
    x, y, w, h = cv2.boundingRect(largest_contour)
    square_size = max(w, h)

    centroid = (x + w // 2, y + h // 2)

    x1 = max(centroid[0] - square_size // 2, 0)
    y1 = max(centroid[1] - square_size // 2, 0)
    x2 = min(centroid[0] + square_size // 2, img.shape[1])
    y2 = min(centroid[1] + square_size // 2, img.shape[0])


    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)

    result = cv2.bitwise_and(img, img, mask=mask)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 255), 2)

    cv2.imshow("ROI", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()