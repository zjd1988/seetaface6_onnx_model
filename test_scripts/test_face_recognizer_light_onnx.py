# -*- coding: utf-8 -*-
import cv2
import onnxruntime
import numpy as np

if __name__ == "__main__":
    onnx_file_name = './models/face_recognizer_light.onnx'
    onnx_session = onnxruntime.InferenceSession(onnx_file_name)
    input_feed = {}
    img = cv2.imread("./test_new.jpg")
    resize_img = cv2.resize(img, (112, 112))
    # resize_img = np.fromfile("E:/github_codes/seetaface_model_parse/seetaface6_onnx_model/recog_input.bin",dtype = np.uint8).reshape((112, 112, 3))
    input_data = (resize_img / 255.0).transpose((2, 0, 1))
    input_feed['_input_data_149'] = input_data.reshape((1, 3, 112, 112)).astype(np.float32)
    pred_result = onnx_session.run([], input_feed=input_feed)
    print(pred_result[0].shape)
    # ori_predict = np.fromfile("E:/github_codes/seetaface_model_parse/seetaface6_onnx_model/recog_output.bin", dtype = np.float32).reshape((1, 512))
    # normalization feature
    norm = pred_result[0] / np.linalg.norm(pred_result[0], axis=1)
    print(norm)

