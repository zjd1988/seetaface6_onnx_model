
import onnx
import time
import numpy as np
from csta_model_parse import parse_csta_model_file
from tsm_to_onnx import construct_onnx_from_tsm


def log_nodes_info(tsm_module_info):
    nodes_info = tsm_module_info['graph']['nodes_info']
    node_op_set = set()
    for index in range(len(nodes_info)):
        node = nodes_info[index]
        if node['bubble_info']['op'] != "<const>":
            node_op_set.add(node['bubble_info']['op'])
            print(index, node)
    
    print(node_op_set)

def conver_csta_model_to_onnx(csta_model_file, onnx_model_file):
    csta_model_info, tsm_module_info, parse_result = parse_csta_model_file(csta_model_file)
    if parse_result:
        # log_nodes_info(tsm_module_info)
        onnx_model, convert_result = construct_onnx_from_tsm(tsm_module_info, log_verbose=True)
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
    csta_file_path = "E:/github_codes/seetaface_model_parse/seetaface6_models/sf3.0_models/face_landmarker_pts5.csta"
    onnx_file_path = "./face_landmarker_pts5.onnx"
    conver_csta_model_to_onnx(csta_file_path, onnx_file_path)