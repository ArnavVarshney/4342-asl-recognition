# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.09.09
""" python demo usage about MNN API """

import sys

import MNN
import cv2


def inference():
    interpreter = MNN.Interpreter("mnn_out/model.mnn")
    interpreter.setCacheFile('.tempcache')
    config = {}
    config['precision'] = 'low'
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    interpreter.resizeTensor(input_tensor, (-1, 26, 26, 1))
    interpreter.resizeSession(session)

    camera = cv2.VideoCapture(0)
    ret, image = camera.read()
    if not ret:
        print("read camera failed")
        sys.exit(1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (26, 26))


if __name__ == "__main__":
    inference()
