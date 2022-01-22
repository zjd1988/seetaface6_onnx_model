import sys
from turtle import shape
import numpy as np
import onnx
import onnxruntime
from onnxsim import simplify

def parse_tensor_shape(tensor_shape_str):
    tensor_shape_dict = {}
    tensor_shape_ele = tensor_shape_str.split(",")
    for shape_ele in tensor_shape_ele:
        tensor_name = shape_ele.split(":")[0]
        shape_str = shape_ele.split(":")[1]
        shape_list = [int(i) for i in shape_str.split("x")]
        tensor_shape_dict[tensor_name] = shape_list
    return tensor_shape_dict

def get_model_info(onnx_file_name):
    onnx_session = onnxruntime.InferenceSession(onnx_file_name)
    input_names = []
    for onnx_node in onnx_session.get_inputs():
        input_names.append(onnx_node.name)

    output_names = []
    for onnx_node in onnx_session.get_outputs():
        output_names.append(onnx_node.name)

    return onnx_session, input_names, output_names


def modify_model_shape(onnx_file_name, input_shapes, output_shapes):
    model = onnx.load(onnx_file_name)
    for i in range(len(model.graph.input)):
        tensor_name = model.graph.input[i].name
        shape_len = len(model.graph.input[i].type.tensor_type.shape.dim)
        for j in range(shape_len):
            model.graph.input[i].type.tensor_type.shape.dim[j].dim_value = input_shapes[tensor_name][j]

    for i in range(len(model.graph.output)):
        tensor_name = model.graph.output[i].name
        shape_len = len(model.graph.output[i].type.tensor_type.shape.dim)
        for j in range(shape_len):
            model.graph.output[i].type.tensor_type.shape.dim[j].dim_value = output_shapes[tensor_name][j]
    return model

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("wrong input parameters....")
        print("{} input_onnx_name tensorname1:shape,tesnsorname2:shape output_onnx_name".format(sys.argv[0]))
        print("for example:{} test.onnx input:1x3x480x480 save.onnx".format(sys.argv[0]))
    input_onnx_name = sys.argv[1]
    tensor_shape_str = sys.argv[2]
    output_onnx_name = sys.argv[3]
    tensor_shape_dict = parse_tensor_shape(tensor_shape_str)
    # 1 prerun model get model info 
    onnx_session, input_names, output_names = get_model_info(input_onnx_name)

    # 2 check model input with sys.argv[2]
    input_tensor_set = set(tensor_shape_dict.keys())
    model_tensor_set = set(input_names)
    if model_tensor_set < input_tensor_set:
        print("model tensor names {} differen with input tensor {}".format(input_tensor_set, model_tensor_set))
    else:
        # 3 inference with new shape get output tensors shape
        model_input_shape_dict = {}
        model_output_shape_dict = {}
        input_feed = {}
        for tensor_name in input_names:
            tensor_shape = tensor_shape_dict[tensor_name]
            input_data = np.ones(tensor_shape).astype(np.float32)
            input_feed[tensor_name] = input_data
        pred_result = onnx_session.run(output_names, input_feed=input_feed)

        # 4 modify model input/output shape
        for i in range(len(output_names)):
            model_output_shape_dict[output_names[i]] = pred_result[i].shape
        for i in range(len(input_names)):
            model_input_shape_dict[input_names[i]] = input_feed[input_names[i]].shape        
        
        new_model = modify_model_shape(input_onnx_name, model_input_shape_dict, model_output_shape_dict)

        # 5 save new model
        model_simp, check = simplify(new_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, output_onnx_name)