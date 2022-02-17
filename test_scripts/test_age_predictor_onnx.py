# -*- coding: utf-8 -*-
import os
import sys
import cv2
import onnxruntime
import numpy as np

if __name__ == "__main__":
    onnx_file_name = './models/age_predictor.onnx'
    onnx_session = onnxruntime.InferenceSession(onnx_file_name)
    input_names = []
    for onnx_node in onnx_session.get_inputs():
        input_names.append(onnx_node.name)
    input_feed = {}
    img = cv2.imread("./test_new.jpg")
    ori_img = img.copy()
    img = cv2.resize(img, (248, 248))
    img_shape = img.shape
    img = img.reshape((1, img_shape[0], img_shape[1], img_shape[2]))
    img = img.transpose((0, 3, 1, 2)).astype(np.float32)
    for item in input_names:
        input_feed[item] = img

    pred_result = onnx_session.run([], input_feed=input_feed)
    print(pred_result)