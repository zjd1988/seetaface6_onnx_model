
import onnx
import time
import numpy as np
from csta_model_parse import parse_csta_model_file
from tsm_to_onnx import construct_onnx_from_tsm, check_unspported_tsm_op


def log_nodes_info(tsm_module_info):
    nodes_info = tsm_module_info['graph']['nodes_info']
    node_op_set = set()
    for index in range(len(nodes_info)):
        node = nodes_info[index]
        if node['bubble_info']['op'] != "<const>":
            node_op_set.add(node['bubble_info']['op'])
            print(index, node)
    print(node_op_set)

def conver_csta_model_to_onnx(csta_model_file, onnx_model_file, input_shape):
    csta_model_info, tsm_module_info, parse_result = parse_csta_model_file(csta_model_file)
    parse_result = parse_result and check_unspported_tsm_op(tsm_module_info)
    if parse_result:
        # log_nodes_info(tsm_module_info)
        onnx_model, convert_result = construct_onnx_from_tsm(tsm_module_info, input_shape, log_verbose=True)
        if convert_result:
            # onnx.helper.printable_graph(onnx_model.graph)
            onnx.save(onnx_model, onnx_model_file)
        else:
            print("convert csta model to onnx fail")
    else:
        print("parse csta model fail")


if __name__ == "__main__":
    # csta_file_path = "E:/github_codes/seetaface_model_parse/seetaface6_models/sf3.0_models/face_detector.csta"
    # onnx_file_path = "./face_detector.onnx"
    # csta_file_path = "E:/github_codes/seetaface_model_parse/seetaface6_models/sf3.0_models/face_landmarker_pts5.csta"
    # csta_file_path = "E:/github_codes/seetaface_model_parse/seetaface6_models/sf3.0_models/face_recognizer_light.csta"
    # onnx_file_path = "./face_recognizer_light.onnx"
    # face_recognizer_light_input_shape = [1, 3, 112, 112]
    # csta_file_path = "E:/github_codes/seetaface_model_parse/seetaface6_models/sf3.0_models/face_recognizer.csta"
    # onnx_file_path = "./face_recognizer.onnx"
    # face_recognizer_light_input_shape = [1, 3, 256, 256]
    # csta_file_path = "E:/github_codes/seetaface_model_parse/seetaface6_models/sf3.0_models/fas_first.csta"
    # onnx_file_path = "./fas_first.onnx"
    # face_recognizer_light_input_shape = [1, 3, 224, 224]
    csta_file_path = "E:/github_codes/seetaface_model_parse/seetaface6_models/sf3.0_models/fas_second.csta"
    onnx_file_path = "./fas_second.onnx"
    face_recognizer_light_input_shape = [1, 3, 300, 300]
    conver_csta_model_to_onnx(csta_file_path, onnx_file_path, face_recognizer_light_input_shape)