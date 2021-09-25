# -*- coding: utf-8 -*-
import os
import sys
import cv2
import onnxruntime
import numpy as np
sys.path.append(os.path.dirname(__file__) + os.sep + "../")
from common.common_util import letterbox
from common.face_detector_util import prior_box_forward, decode, nms_sorted
from common.face_landmark_points_util import shape_index_process
from common.face_anti_spoofing_util import box_detect_postprocess, clarity_estimate

def detect_face(ori_img):
    onnx_file_name = './models/face_detector.onnx'
    onnx_session = onnxruntime.InferenceSession(onnx_file_name)
    input_names = []
    for onnx_node in onnx_session.get_inputs():
        input_names.append(onnx_node.name)
    input_feed = {}
    img = ori_img.copy()
    # preprocess
    resize_img, ratio, _ = letterbox(img, (480, 640), auto=False, scaleFill=True)
    img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB).astype(np.float32)
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
    boxes = decode(pred_result[0], priors, variance, pred_result[1], 0.7, 5000)
    nms_thresh = 0.3
    boxes = nms_sorted(boxes, nms_thresh)

    face_list = []
    for box in boxes:
        box[0] = box[0] * img_shape[1] / ratio[0]
        box[1] = box[1] * img_shape[0] / ratio[1]
        box[2] = box[2] * img_shape[1] / ratio[0]
        box[3] = box[3] * img_shape[0] / ratio[1]
        face_image = ori_img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        face_list.append(face_image)
        # cv2.rectangle(ori_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), point_color, thickness, lineType)
    return face_list, boxes

def face_landmark(ori_img, face_img, face_rect, draw_landmark = False):
    session = onnxruntime.InferenceSession("./models/face_landmarker_pts5_net1.onnx")
    first_input_name = session.get_inputs()[0].name

    resize_img, ratio, _ = letterbox(face_img, (112, 112), auto=False, scaleFill=True)
    # cv2.imwrite("resize.jpg", resize_img)
    gray_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img.reshape((1, 1, 112, 112)).astype(np.float32)
    # points5 net1
    results_1 = session.run([], {first_input_name : gray_img})

    # shape index process
    feat_data = results_1[0]
    pos_data = results_1[1]
    shape_index_results = shape_index_process(feat_data, pos_data)

    # points5 net2
    session = onnxruntime.InferenceSession("./models/face_landmarker_pts5_net2.onnx")
    first_input_name = session.get_inputs()[0].name
    results_2 = session.run([], {first_input_name : shape_index_results})


    landmarks = (results_2[0] + results_1[1])*112
    landmarks = landmarks.reshape((-1)).astype(np.int32)
    ratio_arr = np.array([ratio[index] for index in range(len(ratio))] * 5)
    landmarks = landmarks / ratio_arr
    landmarks = landmarks.astype(np.int32)
    offset_x = int(face_rect[0])
    offset_y = int(face_rect[1])    
    for i in range(landmarks.size // 2):
        landmarks[2*i] += offset_x
        landmarks[2*i + 1] += offset_y
    # 3 draw result
    if draw_landmark == True:
        point_size = 1
        point_color = (0, 0, 255) # BGR
        thickness = 4 # 可以为 0 、4、8
        src_img = ori_img.copy()
        for i in range(landmarks.size // 2):
            point = (landmarks[2*i], landmarks[2*i + 1])
            cv2.circle(src_img, point, point_size, point_color, thickness)

        cv2.imwrite("temp.jpg", src_img)
    return landmarks

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

    # # test
    # img = np.fromfile("./fas_input.bin", dtype=np.float32).reshape((1, 300, 300, 3))
    # img = img.transpose((0, 3, 1, 2)).astype(np.float32)
    # set input
    for item in input_names:
        input_feed[item] = img
    # run session
    pred_result = onnx_session.run([], input_feed=input_feed)
    # post process
    
    # box_encodings = np.fromfile("./fas_output1.bin", dtype=np.float32).reshape((1, 1917, 1, 4))
    # class_predictions = np.fromfile("./fas_output2.bin", dtype=np.float32).reshape((1, 1917, 3, 1))
    class_predictions = pred_result[0]
    box_encodings = pred_result[1]
    return box_detect_postprocess(ori_img, box_encodings, class_predictions)
    # print("run here")

def predict_image(image, face_rect, points):
    # 1 
    boxes = detect_box(image)

    # score face
    face_image = image[int(face_rect[1]):int(face_rect[3]), int(face_rect[0]):int(face_rect[2])]
    clarity = clarity_estimate(face_image)

    passive_result = 0.0
    if len(boxes) == 0:
        crop_face = cv2.resize(face_image, (256, 256))
        start_x = int((256 - 224) / 2)
        end_x   = start_x + 224
        start_y = int((256 - 224) / 2)
        end_y   = start_y + 224
        crop_face = crop_face[start_y:end_y, start_x:end_x]
        crop_face = cv2.cvtColor(crop_face, cv2.COLOR_BGR2YCrCb)
        img_shape = crop_face.shape
        crop_face = crop_face.reshape((1, img_shape[0], img_shape[1], img_shape[2]))
        crop_face = crop_face.transpose((0, 3, 1, 2)).astype(np.float32)
        onnx_file_name = './models/fas_first.onnx'
        onnx_session = onnxruntime.InferenceSession(onnx_file_name)
        input_names = []
        for onnx_node in onnx_session.get_inputs():
            input_names.append(onnx_node.name)
        input_feed = {}
        # set input
        for item in input_names:
            input_feed[item] = crop_face
        # run session
        pred_result = onnx_session.run([], input_feed=input_feed)
        passive_result =  pred_result[0].reshape(-1,)[1]

    fuse_threshold = 0.9
    clarity_threshold = 0.3
    if passive_result >= fuse_threshold:
        if clarity >= clarity_threshold:
            return "REAL"
        else:
            return "FUZZY"
    else:
        return "SPOOF"

if __name__ == "__main__":
    # img = cv2.imread("./test_new.jpg")
    img = cv2.imread("./face_as_test1.jpg")
    
    # 1 detect face
    face_imgs, boxes = detect_face(img)
    assert len(face_imgs) == len(boxes)

    # 2 get face landmark
    # only porcess one face
    src_img = img.copy()
    for index in range(len(face_imgs)):
        landmarks = face_landmark(img, face_imgs[index], boxes[index], draw_landmark=True)

        # 3 predict face image anti spoofing
        predict_result = predict_image(img, boxes[index], landmarks)

        # 4 draw result
        rect_color = (255, 0, 0) # BGR
        point_color = (0, 0, 255) # BGR
        thickness = 4 
        lineType = 4
        point_size = 1
        cv2.rectangle(src_img, (int(boxes[index][0]), int(boxes[index][1])), (int(boxes[index][2]), int(boxes[index][3])), rect_color, thickness, lineType)
        font = cv2.FONT_HERSHEY_SIMPLEX
        imgzi = cv2.putText(src_img, predict_result, (int(boxes[index][0]) - 10, int(boxes[index][1] - 10)), font, 0.8, (255, 255, 0), 2)
        for i in range(landmarks.size // 2):
            point = (landmarks[2*i], landmarks[2*i + 1])
            cv2.circle(src_img, point, point_size, point_color, thickness)
    cv2.imwrite("face_as_image.jpg", src_img)