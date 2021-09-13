# -*- coding: utf-8 -*-
import os
import sys
import cv2
import onnxruntime
import numpy as np
sys.path.append(os.path.dirname(__file__) + os.sep + "../")
from common.face_detector_util import prior_box_forward, decode, nms_sorted

if __name__ == "__main__":
    onnx_file_name = './models/face_detector.onnx'
    onnx_session = onnxruntime.InferenceSession(onnx_file_name)
    input_names = []
    for onnx_node in onnx_session.get_inputs():
        input_names.append(onnx_node.name)
    input_feed = {}
    img = cv2.imread("./test_new.jpg")
    ori_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = img - np.array([123, 117, 104])
    img_shape = img.shape
    img = img.reshape((1, img_shape[0], img_shape[1], img_shape[2]))
    img = img.transpose((0, 3, 1, 2)).astype(np.float32)
    for item in input_names:
        input_feed[item] = img

    pred_result = onnx_session.run([], input_feed=input_feed)

    priors = prior_box_forward(pred_result[2], img_shape[0], img_shape[1], 3)
    variance = [0.1, 0.2]
    boxes = decode(pred_result[0], priors, variance, pred_result[1], 0.9, 5000)
    nms_thresh = 0.3
    boxes = nms_sorted(boxes, nms_thresh)
    for box in boxes:
        box[0] = box[0] * img_shape[1]
        box[1] = box[1] * img_shape[0]
        box[2] = box[2] * img_shape[1]
        box[3] = box[3] * img_shape[0]
        point_color = (0, 255, 0) # BGR
        thickness = 1 
        lineType = 4
        cv2.rectangle(ori_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), point_color, thickness, lineType)
    cv2.imwrite("./face_detect_result.jpg", ori_img)