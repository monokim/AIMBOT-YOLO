import numpy as np
import pyautogui
import win32api, win32con, win32gui
import cv2
import math
import time

CONFIG_FILE = './yolov3.cfg'
WEIGHT_FILE = './yolov3.weights'

net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHT_FILE)
#net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Get rect of Window
hwnd = win32gui.FindWindow(None, 'Counter-Strike: Global Offensive')
rect = win32gui.GetWindowRect(hwnd)
region = rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]

size_scale = 2
while True:
    # Get image of screen
    frame = np.array(pyautogui.screenshot(region=region))
    frame_height, frame_width = frame.shape[:2]

    # Detection
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.7 and classID == 0:
                box = detection[:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.6)

    # Calculate distance for picking the closest enemy from crosshair
    if len(indices) > 0:
        print(f"Detected:{len(indices)}")
        min = 99999
        min_at = 0
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

            dist = math.sqrt(math.pow(frame_width/2 - (x+w/2), 2) + math.pow(frame_height/2 - (y+h/2), 2))
            if dist < min:
                min = dist
                min_at = i

        # Distance of the closest from crosshair
        x = int(boxes[min_at][0] + boxes[min_at][2]/2 - frame_width/2)
        y = int(boxes[min_at][1] + boxes[min_at][3]/2 - frame_height/2) - boxes[min_at][3] * 0.5 # For head shot

        # Move mouse and shoot
        scale = 1.7
        x = int(x * scale)
        y = int(y * scale)
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y, 0, 0)
        time.sleep(0.05)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (frame.shape[1] // size_scale, frame.shape[0] // size_scale))
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
