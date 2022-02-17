# -*- coding: utf-8 -*-
import cv2
import onnxruntime
import numpy as np

if __name__ == "__main__":
    onnx_file_name = './models/face_recognizer.onnx'
    onnx_session = onnxruntime.InferenceSession(onnx_file_name)
    input_feed = {}
    img = cv2.imread("./test_new.jpg")
    resize_img = cv2.resize(img, (248, 248))[..., ::-1]
    input_data = resize_img.transpose((2, 0, 1))
    input_feed['_input_123'] = input_data.reshape((1, 3, 248, 248)).astype(np.float32)
    pred_result = onnx_session.run([], input_feed=input_feed)
    print(pred_result[0].shape)
    # post process
    # 1 sqrt feature
    temp_result = np.sqrt(pred_result[0])
    # 2 normalization feature
    norm = temp_result / np.linalg.norm(temp_result, axis=1)
    print(norm)

