# -*- coding: utf-8 -*-
import os
import sys
import onnxruntime
import numpy as np
import cv2
sys.path.append(os.path.dirname(__file__) + os.sep + "../")
from common.face_landmark_points_util import shape_index_process
from common.face_detector_util import prior_box_forward, decode, nms_sorted

def detect_face(img):
    onnx_file_name = './models/face_detector.onnx'
    onnx_session = onnxruntime.InferenceSession(onnx_file_name)
    input_names = []
    for onnx_node in onnx_session.get_inputs():
        input_names.append(onnx_node.name)
    input_feed = {}
    ori_img = img.copy()
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
    return face_list

def face_landmark(face_img):
    session = onnxruntime.InferenceSession("./models/face_landmarker_pts81_net1.onnx")
    first_input_name = session.get_inputs()[0].name

    gray_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img.reshape((1, 1, 112, 112)).astype(np.float32)
    # points5 net1
    results_1 = session.run([], {first_input_name : gray_img})

    # shape index process
    feat_data = results_1[0]
    pos_data = results_1[1]
    shape_index_results = shape_index_process(feat_data, pos_data)

    # points5 net2
    session = onnxruntime.InferenceSession("./models/face_landmarker_pts81_net2.onnx")
    first_input_name = session.get_inputs()[0].name
    results_2 = session.run([], {first_input_name : shape_index_results})

    landmarks = (results_2[0] + results_1[1])*112
    landmarks = landmarks.reshape((-1)).astype(np.int32)

    return landmarks

if __name__ == "__main__":
    img = cv2.imread("./test_new.jpg")
    # 1. detect face
    crop_face = detect_face(img)

    # 2. get face 81 point landmark
    if len(crop_face) > 0:
        crop_face[0] = cv2.resize(crop_face[0], (112, 112))
        landmarks = face_landmark(crop_face[0])

        # 3 draw result
        point_size = 1
        point_color = (0, 0, 255) # BGR
        thickness = 4 # 可以为 0 、4、8
        for i in range(landmarks.size // 2):
            point = (landmarks[2*i], landmarks[2*i + 1])
            cv2.circle(crop_face[0], point, point_size, point_color, thickness)

        cv2.imwrite("face_landmard_81_points_result.jpg", crop_face[0])
        print(landmarks)