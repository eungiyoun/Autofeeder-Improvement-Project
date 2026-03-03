#onnx_detector.py
import cv2
import numpy as np
import onnxruntime as ort
import time
import pandas as pd
import copy

from detector import ONNXDetector

MODEL_PATH = './model/best_int8.onnx'
IMG_PATH = './test/img.jpg'

model = ONNXDetector(MODEL_PATH)
img = cv2.imread(IMG_PATH)

detections = model.detect(img, conf_thres=0.25, iou_thres=0.45)
print("NUM DETS:", len(detections))

img_out = model.visualize(img, detections)
cv2.imwrite("result.jpg", img_out)

# import cv2
# from detector.yolo_detector import YOLODetector

# MODEL_PATH = "./model/best.pt"
# IMG_PATH = "./image.jpg"

# # Load model
# model = YOLODetector(MODEL_PATH, conf_threshold=0.25)

# # Load image
# img = cv2.imread(IMG_PATH)

# # Detection (same output format as ONNXDetector)
# # returns: [x1, y1, x2, y2, score, class_id]
# detections = model.detect(img)
# print("NUM DETS:", len(detections))

# # Visualization
# img_out = model.visualize(img, detections)
# cv2.imwrite("yolo_result.jpg", img_out)

# print("Saved → yolo_result.jpg")
