import cv2
import numpy as np
from ultralytics import YOLO
import os

CLASS_NAMES = ["DIMPLE", "INNER CENTER", "LOBE"]
COLOR_PALETTE = [(255, 0, 0), (0, 255, 0),  (0, 0, 255)]


class YOLODetector:
    def __init__(self, model_path, conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.model.to("cpu")     # force CPU
        self.conf_threshold = conf_threshold


    def infer(self, image):
        return self.model(image, device="cpu", verbose=False)[0]


    def detect(self, image):
        result = self.infer(image)

        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
        class_ids = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else []

        if len(boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        # Confidence filtering
        mask = confs >= self.conf_threshold
        boxes = boxes[mask]
        confs = confs[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        detections = []
        for box, score, cid in zip(boxes, confs, class_ids):
            detections.append([
                float(box[0]), float(box[1]),
                float(box[2]), float(box[3]),
                float(score),
                int(cid)
            ])

        return np.array(detections, dtype=np.float32)


    def visualize(self, image, detections):
        img = image.copy()

        for det in detections:
            x1, y1, x2, y2, score, class_id = det
            class_id = int(class_id)

            color = COLOR_PALETTE[class_id]
            label = f"{CLASS_NAMES[class_id]} {score:.2f}"

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, 0, 0.6, 1)

            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(
                img, label, (x1, y1 - 2),
                0, 0.6, (255, 255, 255), 1
            )

        return img
