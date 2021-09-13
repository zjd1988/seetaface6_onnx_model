# -*- coding: utf-8 -*-
import cv2
import onnxruntime
import numpy as np
from common.face_detector_util import prior_box_forward, decode, nms_sorted

def detect_face(ori_img):
    onnx_file_name = './models/face_detector.onnx'
    onnx_session = onnxruntime.InferenceSession(onnx_file_name)
    input_names = []
    for onnx_node in onnx_session.get_inputs():
        input_names.append(onnx_node.name)
    input_feed = {}
    img = ori_img.copy()
    # preprocess
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = img - np.array([123, 117, 104])
    img_shape = img.shape
    img = img.reshape((1, img_shape[0], img_shape[1], img_shape[2]))
    img = img.transpose((0, 3, 1, 2)).astype(np.float32)
    # set input
    for item in input_names:
        input_feed[item] = img
    # run session
    pred_result = onnx_session.run([], input_feed=input_feed)
    # post process
    priors = prior_box_forward(pred_result[2], img_shape[0], img_shape[1], 3)
    variance = [0.1, 0.2]
    boxes = decode(pred_result[0], priors, variance, pred_result[1], 0.9, 5000)
    nms_thresh = 0.3
    boxes = nms_sorted(boxes, nms_thresh)

    face_list = []
    for box in boxes:
        box[0] = box[0] * img_shape[1]
        box[1] = box[1] * img_shape[0]
        box[2] = box[2] * img_shape[1]
        box[3] = box[3] * img_shape[0]
        face_image = ori_img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        face_list.append(face_image)
        # cv2.rectangle(ori_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), point_color, thickness, lineType)
    return face_list, boxes

def detect_box(ori_img):
    onnx_file_name = './models/fas_second.onnx'
    onnx_session = onnxruntime.InferenceSession(onnx_file_name)
    input_names = []
    for onnx_node in onnx_session.get_inputs():
        input_names.append(onnx_node.name)
    input_feed = {}
    img = ori_img.copy()
    # preprocess
    img = cv2.resize(img, (300, 300))
    img = img.astype(np.float32)
    img = img / 128.0
    img = img - 1.0
    img_shape = img.shape
    img = img.reshape((1, img_shape[0], img_shape[1], img_shape[2]))
    img = img.transpose((0, 3, 1, 2)).astype(np.float32)
    # set input
    for item in input_names:
        input_feed[item] = img
    # run session
    pred_result = onnx_session.run([], input_feed=input_feed)
    # post process


if __name__ == "__main__":
    img = cv2.imread("./test_new.jpg")
    # 1 detect face
    crop_face, boxes = detect_face(img)
    
    # 2 get face landmark
    detect_box(img)