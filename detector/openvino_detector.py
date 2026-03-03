import cv2
import numpy as np
import time
from openvino.runtime import Core

CLASS_NAMES = ["DIMPLE", "INNER CENTER", "LOBE"]
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


class OpenVINODetector:
    def __init__(self, model_path, input_size=640):
        self.core = Core()

        print(f"[OpenVINO] Loading model: {model_path}")
        self.model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(self.model, "CPU")

        self.input_layer = self.compiled_model.inputs[0]
        self.output_layer = self.compiled_model.outputs[0]

        self.input_size = input_size


    # ---------------------------------------------------
    # Preprocess
    # ---------------------------------------------------
    def preprocess(self, image):
        orig_h, orig_w = image.shape[:2]

        img = cv2.resize(image, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None]  # (1,3,H,W)

        return img, (orig_h, orig_w)


    # ---------------------------------------------------
    # Infer
    # ---------------------------------------------------
    def infer(self, tensor):
        result = self.compiled_model([tensor])[self.output_layer]
        return result  # shape: (1, 7, N)


    # ---------------------------------------------------
    # xywh → xyxy
    # ---------------------------------------------------
    def xywh_to_xyxy(self, xywh):
        xyxy = np.zeros_like(xywh)
        xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
        return xyxy


    # ---------------------------------------------------
    # NMS (class-wise)
    # ---------------------------------------------------
    def nms(self, boxes, scores, class_ids, iou_thres):
        keep = []

        for cls in np.unique(class_ids):
            idxs = np.where(class_ids == cls)[0]
            cls_boxes = boxes[idxs]
            cls_scores = scores[idxs]

            order = np.argsort(-cls_scores)

            while len(order) > 0:
                i = order[0]
                keep.append(idxs[i])

                xx1 = np.maximum(cls_boxes[i, 0], cls_boxes[order[1:], 0])
                yy1 = np.maximum(cls_boxes[i, 1], cls_boxes[order[1:], 1])
                xx2 = np.minimum(cls_boxes[i, 2], cls_boxes[order[1:], 2])
                yy2 = np.minimum(cls_boxes[i, 3], cls_boxes[order[1:], 3])

                inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
                area_i = (cls_boxes[i, 2] - cls_boxes[i, 0]) * (cls_boxes[i, 3] - cls_boxes[i, 1])
                area_j = (cls_boxes[order[1:], 2] - cls_boxes[order[1:], 0]) * \
                         (cls_boxes[order[1:], 3] - cls_boxes[order[1:], 1])
                union = area_i + area_j - inter

                iou = inter / (union + 1e-6)
                remain = np.where(iou <= iou_thres)[0]
                order = order[remain + 1]

        return boxes[keep], scores[keep], class_ids[keep]


    # ---------------------------------------------------
    # Postprocess
    # ---------------------------------------------------
    def postprocess(self, raw, orig_shape, conf_thres=0.25, iou_thres=0.45):
        orig_h, orig_w = orig_shape

        raw = raw[0]  # (7, N)
        if raw.ndim != 2 or raw.shape[0] < 7:
            return np.empty((0, 6), dtype=np.float32)

        b_xywh = raw[0:4].T           # (N,4)
        cls_scores = raw[4:].T        # (N,3)

        confidences = np.max(cls_scores, axis=1)
        class_ids = np.argmax(cls_scores, axis=1)

        mask = confidences >= conf_thres
        if not mask.any():
            return np.empty((0, 6), dtype=np.float32)

        b_xywh = b_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        b_xyxy = self.xywh_to_xyxy(b_xywh)

        b_xyxy[:, [0, 2]] *= orig_w / self.input_size
        b_xyxy[:, [1, 3]] *= orig_h / self.input_size

        b_xyxy, confidences, class_ids = self.nms(
            b_xyxy, confidences, class_ids, iou_thres
        )

        return np.column_stack([b_xyxy, confidences, class_ids])


    # ---------------------------------------------------
    # Full detection pipeline
    # ---------------------------------------------------
    def detect(self, image, conf_thres=0.25, iou_thres=0.45):
        tensor, orig_shape = self.preprocess(image)
        raw = self.infer(tensor)
        return self.postprocess(raw, orig_shape, conf_thres, iou_thres)


    # ---------------------------------------------------
    # Visualization
    # ---------------------------------------------------
    def visualize(self, image, detections):
        img = image.copy()
        h, w = img.shape[:2]

        scale = max(0.5, (h + w) / 1000)
        thickness = max(1, int(2 * scale))
        font_scale = 0.5 * scale
        margin = max(2, int(4 * scale))

        for det in detections:
            x1, y1, x2, y2, score, cls_id = det
            cls_id = int(cls_id)
            color = COLORS[cls_id]
            label = f"{CLASS_NAMES[cls_id]} {score:.2f}"

            x1 = int(x1); y1 = int(y1)
            x2 = int(x2); y2 = int(y2)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          font_scale, thickness)

            cv2.rectangle(
                img,
                (x1, y1 - th - 2 * margin),
                (x1 + tw + 2 * margin, y1),
                color,
                -1
            )

            cv2.putText(
                img,
                label,
                (x1 + margin, y1 - margin),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )
        return img
