import math
from numpy.matrixlib.defmatrix import matrix
import onnx
import copy
import time
import onnxruntime
import numpy as np
from onnx import checker
from onnx import helper
from onnx import TensorProto

net_input_shape = [-1, -1, -1, -1]

def get_initializer(nodes_info, initializer_list, initializer_name_dict):
    for i in range(len(nodes_info)):
        node = nodes_info[i]['bubble_info']
        if node['op'] == '<const>' and len(nodes_info[i]['output_index']) > 0:
            # print(i, node['name'])
            # replace int32 to int64
            if node['value'].dtype == np.int32:
                node['value'] = node['value'].astype(np.int64)
            initializer_list.append(i)
            if node['name'] in initializer_name_dict.keys():
                repeat_node_index = initializer_name_dict[node['name']]
                print("already exists initializer {}:{}".format(node['name'], repeat_node_index))
                print("repeat node is {}".format(nodes_info[repeat_node_index]['bubble_info']))
                print("current node is {}".format(node))
            else:
                initializer_name_dict[node['name']] = i
    # return len(initializer_list) == len(initializer_name_dict)
    return True


def conver_numpy_type_to_onnx_type(num_arr):
    datatype = num_arr.dtype
    if datatype == np.int8:
        return TensorProto.INT8
    elif datatype == np.uint8:
        return TensorProto.UINT8
    elif datatype == np.int16:
        return TensorProto.INT16
    elif datatype == np.uint16:
        return TensorProto.UINT16
    elif datatype == np.int32:
        return TensorProto.INT32
    elif datatype == np.uint32:
        return TensorProto.UINT32
    elif datatype == np.int64:
        return TensorProto.INT64
    elif datatype == np.uint64:
        return TensorProto.UINT64
    elif datatype == np.bool:
        return TensorProto.BOOL
    elif datatype == np.float32:
        return TensorProto.FLOAT
    elif datatype == np.float64:
        return TensorProto.DOUBLE
    else:
        print("unspoorted datatype {}".format(datatype))
        return None

# INT8        = 1,
# UINT8       = 2,
# INT16       = 3,
# UINT16      = 4,
# INT32       = 5,
# UINT32      = 6,
# INT64       = 7,
# UINT64      = 8,
# FLOAT16     = 9,
# FLOAT32     = 10,
# FLOAT64     = 11,
def convert_tsm_type_to_onnx_type(tsm_type):
    if tsm_type == 1:
        return 'INT8'        
    elif tsm_type == 2:
        return 'UINT8'
    elif tsm_type == 3:
        return 'INT16'
    elif tsm_type == 4:
        return 'UINT16'
    elif tsm_type == 5:
        return 'INT32'
    elif tsm_type == 6:
        return 'UINT32'
    elif tsm_type == 7:
        return 'INT64'
    elif tsm_type == 8:
        return 'UINT64'
    elif tsm_type == 9:
        return 'FLOAT16'
    elif tsm_type == 10:
        return "FLOAT"
    elif tsm_type == 11:
        return 'DOUBLE'
    else:
        return ""

def convert_onnx_type_to_value(onnx_type):
    if onnx_type == "FLOAT":
        return 1
    elif onnx_type == 'UINT8':
        return 2
    elif onnx_type == 'INT8':
        return 3
    elif onnx_type == 'UINT16':
        return 4
    elif onnx_type == 'INT16':
        return 5
    elif onnx_type == 'INT32':
        return 6
    elif onnx_type == 'INT64':
        return 7
    elif onnx_type == 'BOOL':
        return 9
    elif onnx_type == 'FLOAT16':
        return 10
    elif onnx_type == 'DOUBLE':
        return 11
    elif onnx_type == 'UINT32':
        return 12
    elif onnx_type == 'UINT64':
        return 13
    else:
        return -1

def topo_sort(nodes_info, input_index_list, output_index_list, const_node_list):
    topo_order_list = []
    input_list = copy.deepcopy(input_index_list)
    traversal_node_set = set()
    traversal_node_set.update(const_node_list)
    traversal_node_set.update(output_index_list)
    while len(input_list) > 0:
        input_index = input_list[0]
        if input_index not in traversal_node_set:
            traversal_node_set.add(input_index)
        else:
            del input_list[0]
            continue
        del input_list[0]
        out_list = nodes_info[input_index]['output_index']

        for item in out_list:
            temp_input_list = nodes_info[item]['input_index']
            flag = True
            for temp_item in temp_input_list:
                if temp_item not in traversal_node_set:
                    flag = False
            if flag == True:
                input_list.append(item)
        # input_list.extend(out_list)
        if input_index not in input_index_list and input_index not in output_index_list\
            and input_index not in const_node_list and input_index not in topo_order_list:
            
            topo_order_list.append(input_index)
        if input_index in topo_order_list:
            topo_order_list.remove(input_index)
            topo_order_list.append(input_index)
    # check again
    check_flag = True
    node_index_set = set()
    for index in topo_order_list:
        input_list = nodes_info[index]['input_index']
        for item in input_list:
            if item not in input_index_list and item not in const_node_list and item not in node_index_set:
                check_flag = False
                print("check fail")
                break
    
        node_index_set.add(index)
    return topo_order_list

def get_topo_order(nodes_info):
    input_index_list = []
    output_index_list = []
    const_node_list = []
    remove_node_list = []
    const_node_name_dict = {}
    get_initializer(nodes_info, const_node_list, const_node_name_dict)
    node_index_list = [0 for i in range(len(nodes_info))]
    for i in range(len(nodes_info)):
        node = nodes_info[i]['bubble_info']
        node_input_index_list = nodes_info[i]['input_index']
        node_output_index_list = nodes_info[i]['output_index']
        if len(node_input_index_list) == 0 and len(node_output_index_list) == 0:
            remove_node_list.append(i)
            node_index_list[i] = -1
        elif len(node_input_index_list) == 0 and i not in const_node_list and node['name'] not in const_node_name_dict:
            input_index_list.append(i)
        else:
            for index in node_input_index_list:
                node_index_list[index] = 1
    output_index_list = np.where(np.array(node_index_list) == 0)[0].tolist()

    topo_order_list = topo_sort(nodes_info, input_index_list, output_index_list, const_node_list)
    
    return input_index_list, output_index_list, topo_order_list


class ConstructOnnxOp(object):
    op_construct_funcs = {}
    tsm_op_onnx_op_map = {}
    def register(self, tsm_op_name, onnx_op_name, op_func):
        if tsm_op_name not in self.tsm_op_onnx_op_map.keys():
            self.tsm_op_onnx_op_map[tsm_op_name] = onnx_op_name
            self.op_construct_funcs[onnx_op_name] = op_func
        else:
            print("already register {}".format(tsm_op_name))
            
    def construct_node(self, tsm_nodes_info, node_index, onnx_nodes_info, log_flag = False):
        node_info = tsm_nodes_info[node_index]
        tsm_op_name = node_info['bubble_info']['op']
        if tsm_op_name not in self.tsm_op_onnx_op_map.keys():
            if len(node_info['input_index']) == 0:
                tsm_op_name = "input"
            elif len(node_info['output_index']) == 0:
                tsm_op_name = "output"
            else:
                tsm_op_name = ""
        if tsm_op_name == "":
            tsm_op_name = node_info['bubble_info']['op']
            print("have not register {}".format(tsm_op_name))
            return False
        else:
            onnx_op_name = self.tsm_op_onnx_op_map[tsm_op_name]
            tsm_op_name = node_info['bubble_info']['op']
            if log_flag:
                print("{}: --------------- convert {} to {}".format(node_index, tsm_op_name, onnx_op_name))
                node_input_list = node_info['input_index']
                node_output_list = node_info['output_index']
                for index in range(len(node_input_list)):
                    input_index = node_input_list[index]
                    print("input {}: {}->{}".format(index, input_index, tsm_nodes_info[input_index]['bubble_info']['name']))
                for index in range(len(node_output_list)):
                    output_index = node_output_list[index]
                    print("output {}: {}->{}".format(index, output_index, tsm_nodes_info[output_index]['bubble_info']['name']))

            convert_flag = self.op_construct_funcs[onnx_op_name](tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag)
            if len(node_info['output_index']) == 0:
                onnx_op_name = 'Output'
                if convert_flag == True:
                    convert_flag = self.op_construct_funcs[onnx_op_name](tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag)
            return convert_flag


def get_node_input_output(nodes_info, node_index):
    node_info = nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list = node_info['input_index']
    node_output_list = node_info['output_index']
    input_names = []
    for index in node_input_list:
        input_names.append(nodes_info[index]['bubble_info']['name'])
    output_names = [node['name'], ]

    return node_input_list, node_output_list, input_names, output_names


def infer_shape_dims(onnx_op_name, tsm_nodes_info, node_index):
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list, node_output_list, input_names, output_names = get_node_input_output(tsm_nodes_info, node_index)    
    if onnx_op_name == "Pooling":
        node['shape_dims'] = tsm_nodes_info[node_input_list[0]]['bubble_info']['shape_dims']
    elif onnx_op_name == "Concat":
        if tsm_nodes_info[node_input_list[0]]['bubble_info']['shape_dims'] == 1:
            node['shape_dims'] = 0
            for index in node_input_list:
                node['shape_dims'] += tsm_nodes_info[index]['bubble_info']['shape_dims_value']
        else:
            node['shape_dims'] = tsm_nodes_info[node_input_list[0]]['bubble_info']['shape_dims']
    elif onnx_op_name == "Reshape":
        if len(node_input_list) == 1:
            shape_value = node['shape']
            tensor_shape = list(shape_value.shape)
            assert len(tensor_shape) == 1
            # node['shape_dims_value'] = tensor_shape[0]
            node['shape_dims'] = tensor_shape[0]
        else:
            if tsm_nodes_info[node_input_list[1]]['bubble_info']['shape_dims'] == 1:
                node['shape_dims'] = tsm_nodes_info[node_input_list[1]]['bubble_info']['shape_dims_value']
            else:
                node['shape_dims'] = tsm_nodes_info[node_input_list[1]]['bubble_info']['shape_dims']
    elif onnx_op_name == "Flatten":
        if 'dim' in node.keys():
            assert len(node['dim']) == 1
            axis = node['dim'][0]
        else:
            axis = 1
        if axis == 0:
            node['shape_dims'] = 2
        else:
            node['shape_dims'] = axis + 1
    elif onnx_op_name == "Transpose":
        perm = node['permute'].tolist()
        node['shape_dims'] = len(perm)
    elif onnx_op_name == "Softmax":
        node['shape_dims'] = tsm_nodes_info[node_input_list[0]]['bubble_info']['shape_dims']
    elif onnx_op_name == "Conv" or onnx_op_name == "DepthWiseConv":
        node['shape_dims'] = tsm_nodes_info[node_input_list[0]]['bubble_info']['shape_dims']
    elif onnx_op_name == "Gather":
        node['shape_dims'] = tsm_nodes_info[node_input_list[1]]['bubble_info']['shape_dims'] + tsm_nodes_info[node_input_list[0]]['bubble_info']['shape_dims'] - 1
        if tsm_nodes_info[node_input_list[1]]['bubble_info']['shape_dims'] == 1:
            node['shape_dims_value'] = tsm_nodes_info[node_input_list[1]]['bubble_info']['shape_dims_value']
    # elif onnx_op_name == "ConcatFromSequence":
    #     node['shape_dims'] = tsm_nodes_info[node_input_list[0]]['bubble_info']['shape_dims'] + 1
    elif onnx_op_name == "Unsqueeze":
        node['shape_dims'] = tsm_nodes_info[node_input_list[0]]['bubble_info']['shape_dims'] + 1
    elif onnx_op_name == "BatchNormalization":
        node['shape_dims'] = tsm_nodes_info[node_input_list[0]]['bubble_info']['shape_dims']
    elif onnx_op_name == "Cast":
        node['shape_dims'] = tsm_nodes_info[node_input_list[0]]['bubble_info']['shape_dims']
    elif onnx_op_name == "Sub":
        node['shape_dims'] = tsm_nodes_info[node_input_list[0]]['bubble_info']['shape_dims']
    elif onnx_op_name == "Mul":
        node['shape_dims'] = tsm_nodes_info[node_input_list[0]]['bubble_info']['shape_dims']
    elif onnx_op_name == "Sigmoid":
        node['shape_dims'] = tsm_nodes_info[node_input_list[0]]['bubble_info']['shape_dims']
    elif onnx_op_name == "Relu":
        node['shape_dims'] = tsm_nodes_info[node_input_list[0]]['bubble_info']['shape_dims']
    elif onnx_op_name == "Shape":
        node['shape_dims'] = 1
        node['shape_dims_value'] = tsm_nodes_info[node_input_list[0]]['bubble_info']['shape_dims']
    elif onnx_op_name == "Gemm":
        node['shape_dims'] = 2
    elif onnx_op_name == "Clip":
        node['shape_dims'] = tsm_nodes_info[node_input_list[0]]['bubble_info']['shape_dims']
    elif onnx_op_name == "Add":
        assert len(node_input_list) == 2
        first_input_dims = tsm_nodes_info[node_input_list[0]]['bubble_info']['shape_dims']
        second_input_dims = tsm_nodes_info[node_input_list[1]]['bubble_info']['shape_dims']
        node['shape_dims'] = max(first_input_dims, second_input_dims)
    else:
        print("unpoorted op type {}".format(onnx_op_name))
    print("*******************node {}: shape_dims is {}".format(node_index, node['shape_dims']))


def construct_input_op(tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag = False):
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list, node_output_list, input_names, output_names = get_node_input_output(tsm_nodes_info, node_index)
    assert len(node_input_list) == 0
    init_tensor_name = node['name']
    if 'value' not in node.keys():
        init_tensor_shape = [-1, -1, -1, -1]
        init_tensor_type = TensorProto.FLOAT
    else:
        init_tensor_type = conver_numpy_type_to_onnx_type(node['value'])
        init_tensor_shape = list(node['value'].shape)
        
    if log_flag:
        print("construct tensor value info with:")
        print("tensor name: {}".format(init_tensor_name))
        print("tensor shape: {}".format(init_tensor_shape))
        print("tensor type: {}".format(init_tensor_type))

    node['shape_dims'] = len(init_tensor_shape)
    node_def = helper.make_tensor_value_info(init_tensor_name, init_tensor_type, init_tensor_shape)
    onnx_nodes_info['input'].append(node_def)
    return True


def construct_output_op(tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag = False):
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list, node_output_list, input_names, output_names = get_node_input_output(tsm_nodes_info, node_index)    
    assert len(node_output_list) == 0
    init_tensor_name = node['name']
    # assert 'shape_dims' in node.keys()
    if 'value' not in node.keys():
        # init_tensor_shape = [-1 for i in range(node['shape_dims'])]        
        if 'shape_dims' in node.keys():
            init_tensor_shape = [-1 for i in range(node['shape_dims'])]
        else:
            input_node = tsm_nodes_info[node_input_list[0]]['bubble_info']
            init_tensor_shape = [-1 for i in range(input_node['shape_dims'])]
        init_tensor_type = TensorProto.FLOAT
    else:
        init_tensor_type = conver_numpy_type_to_onnx_type(node['value'])
        init_tensor_shape = list(node['value'].shape)
    if log_flag:
        print("construct tensor value info with:")
        print("tensor name: {}".format(init_tensor_name))
        print("tensor shape: {}".format(init_tensor_shape))
        print("tensor type: {}".format(init_tensor_type))
    node_def = helper.make_tensor_value_info(init_tensor_name, init_tensor_type, init_tensor_shape)
    onnx_nodes_info['output'].append(node_def)
    return True


def construct_const_op(tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag = False):
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list = node_info['input_index']
    assert len(node_input_list) == 0
    init_tensor_name = node['name']
    init_tensor_type = conver_numpy_type_to_onnx_type(node['value'])
    init_tensor_shape = list(node['value'].shape)
    init_tensor_data = node['value'].copy()
    init_tensor_data = init_tensor_data.reshape(1,-1)[0]
    if log_flag:
        print("construct const tensor with:")
        print("tensor name: {}".format(init_tensor_name))
        print("tensor shape: {}".format(init_tensor_shape))
        print("tensor type: {}".format(init_tensor_type))
        print("tensor data: {}".format(init_tensor_data))

    node['shape_dims'] = len(init_tensor_shape)
    if len(init_tensor_shape) == 1:
        node['shape_dims_value'] = init_tensor_shape[0]
    node_def = helper.make_tensor(init_tensor_name, init_tensor_type, init_tensor_shape, init_tensor_data)
    onnx_nodes_info['initializer'].append(node_def)
    return True


def compute_onnx_pad_value(tsm_nodes_info, node_index, onnx_nodes_info):
    global net_input_shape
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list = node_info['input_index']
    init_tensor_name = tsm_nodes_info[node_input_list[0]]['bubble_info']['name']
    init_tensor_shape = [-1, -1, -1, -1]
    # init_tensor_shape = net_input_shape
    init_tensor_type = TensorProto.FLOAT
    output_tensor = helper.make_tensor_value_info(init_tensor_name, init_tensor_type, init_tensor_shape)
    onnx_initializer_list = onnx_nodes_info['initializer']
    onnx_input_list = onnx_nodes_info['input']
    onnx_output_list = [output_tensor, ]
    onnx_node_list = onnx_nodes_info['node']
    onnx_graph_def = helper.make_graph(onnx_node_list, 'pooling_pad_graph', onnx_input_list, onnx_output_list, initializer=onnx_initializer_list)
    op = onnx.OperatorSetIdProto()
    op.version = 11
    onnx_mode_def = helper.make_model(onnx_graph_def, producer_name='csta_parser', opset_imports=[op])
    onnx_mode_def.ir_version = 6
    onnx.checker.check_model(onnx_mode_def)
    onnx_file_name = "./temp.onnx"
    onnx.save(onnx_mode_def, onnx_file_name)
    onnx_session = onnxruntime.InferenceSession(onnx_file_name)
    input_names = []
    for onnx_node in onnx_session.get_inputs():
        input_names.append(onnx_node.name)
    input_feed = {}
    for item in input_names:
        # input_feed[item] = np.zeros((1,3,480,640)).astype(np.float32)
        input_feed[item] = np.zeros(init_tensor_shape).astype(np.float32)
    
    pred_result = onnx_session.run([], input_feed=input_feed)
    input_size = pred_result[0].shape[2:]
    ksize = tsm_nodes_info[node_input_list[1]]['bubble_info']['value'].tolist()[2:]
    stride = tsm_nodes_info[node_input_list[2]]['bubble_info']['value'].tolist()[2:]
    if 'padding' in node.keys():
        static_padding = node['padding'][2:,:].transpose((1,0)).reshape((1,-1))[0].tolist()
    else:
        static_padding = [0, 0, 0, 0]
    dynamic_padding = [0 for i in range(4)]
    dynamic_padding[0] = static_padding[0]
    dynamic_padding[1] = static_padding[1]
    dynamic_padding[2] = static_padding[2] - (input_size[0] + static_padding[0] + static_padding[2] - ksize[0]) % stride[0]
    dynamic_padding[3] = static_padding[3] - (input_size[1] + static_padding[1] + static_padding[3] - ksize[1]) % stride[1]
    # print("run here")
    return dynamic_padding


def construct_pooling_pad_op(tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag = False):
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list, node_output_list, input_names, output_names = get_node_input_output(tsm_nodes_info, node_index)
    assert len(node_input_list) == 3
    assert len(node_output_list) == 1
    auto_pad = node['auto_pad']
    out_index = node_output_list[0]
    assert node_index in tsm_nodes_info[out_index]['input_index']
    if log_flag:
        print("construct pooling_pad with:")
        print("auto_pad: {}".format(auto_pad))
        print("out_index: {}".format(out_index))
    tsm_nodes_info[out_index]['bubble_info']['auto_pad'] = auto_pad
    tsm_nodes_info[out_index]['bubble_info']['pads'] = compute_onnx_pad_value(tsm_nodes_info, node_index, onnx_nodes_info)
    return True


def compute_tf_pad_value(tsm_nodes_info, node_index, onnx_nodes_info):
    global net_input_shape
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list = node_info['input_index']
    init_tensor_name = tsm_nodes_info[node_input_list[0]]['bubble_info']['name']
    init_tensor_shape = [-1, -1, -1, -1]
    # init_tensor_shape = net_input_shape
    init_tensor_type = TensorProto.FLOAT
    output_tensor = helper.make_tensor_value_info(init_tensor_name, init_tensor_type, init_tensor_shape)
    onnx_initializer_list = onnx_nodes_info['initializer']
    onnx_input_list = onnx_nodes_info['input']
    onnx_output_list = [output_tensor, ]
    onnx_node_list = onnx_nodes_info['node']
    onnx_graph_def = helper.make_graph(onnx_node_list, 'tf_conv2d_pad_graph', onnx_input_list, onnx_output_list, initializer=onnx_initializer_list)
    op = onnx.OperatorSetIdProto()
    op.version = 11
    onnx_mode_def = helper.make_model(onnx_graph_def, producer_name='csta_parser', opset_imports=[op])
    onnx_mode_def.ir_version = 6
    onnx.checker.check_model(onnx_mode_def)
    onnx_file_name = "./temp.onnx"
    onnx.save(onnx_mode_def, onnx_file_name)
    onnx_session = onnxruntime.InferenceSession(onnx_file_name)
    input_names = []
    for onnx_node in onnx_session.get_inputs():
        input_names.append(onnx_node.name)
    input_feed = {}
    for item in input_names:
        # input_feed[item] = np.zeros((1,3,480,640)).astype(np.float32)
        input_feed[item] = np.zeros(net_input_shape).astype(np.float32)
    
    pred_result = onnx_session.run([], input_feed=input_feed)
    input_size = pred_result[0].shape[2:]
    ksize = tsm_nodes_info[node_input_list[1]]['bubble_info']['value'].shape[2:]
    stride = node['stride'].tolist()[2:]
    dilation = node['dilation'].tolist()[2:]
    if 'padding' in node.keys():
        static_padding = node['padding'][2:,:].transpose((1,0)).reshape((1,-1))[0].tolist()
    else:
        static_padding = [0, 0, 0, 0]
    
    expect_output_size = [0, 0]
    expect_input_size = [0, 0]
    dynamic_padding = [0 for i in range(4)]
    if node['padding_method'] == 'VALID':
        this_kernel_height    = (ksize[0] - 1) * dilation[0] + 1
        this_kernel_width     = (ksize[1] - 1) * dilation[1] + 1
        expect_output_size[0] = int(math.ceil((input_size[0] + static_padding[0] + static_padding[1]  - this_kernel_height + 1) / float(stride[0])))
        expect_output_size[1] = int(math.ceil((input_size[1] + static_padding[2] + static_padding[3]  - this_kernel_width + 1) / float(stride[1])))
        expect_input_size[0]  = (expect_output_size[0] - 1) * stride[0] + (dilation[0] * (ksize[0] - 1) + 1) - static_padding[0] - static_padding[1]
        expect_input_size[1]  = (expect_output_size[1] - 1) * stride[1] + (dilation[1] * (ksize[1] - 1) + 1) - static_padding[2] - static_padding[3]
        dynamic_padding[0]    = static_padding[0]
        dynamic_padding[2]    = static_padding[1] + expect_input_size[0] - input_size[0]
        dynamic_padding[1]    = static_padding[2]
        dynamic_padding[3]    = static_padding[3] + expect_input_size[1] - input_size[1]
    else:
        this_kernel_height    = (ksize[0] - 1) * dilation[0] + 1
        this_kernel_width     = (ksize[1] - 1) * dilation[1] + 1
        expect_output_size[0] = int(math.ceil((input_size[0] + static_padding[0] + static_padding[1]) / float(stride[0])))
        expect_output_size[1] = int(math.ceil((input_size[1] + static_padding[2] + static_padding[3]) / float(stride[1])))
        expect_input_size[0]  = (expect_output_size[0] - 1) * stride[0] + (dilation[0] * (ksize[0] - 1) + 1) - static_padding[0] - static_padding[1]
        expect_input_size[1]  = (expect_output_size[1] - 1) * stride[1] + (dilation[1] * (ksize[1] - 1) + 1) - static_padding[2] - static_padding[3]
        padding_height        = (expect_input_size[0] - input_size[0])
        padding_width         = (expect_input_size[1] - input_size[1])
        half_padding_height   = int(padding_height / 2)
        half_padding_width    = int(padding_width / 2)
        dynamic_padding[0]    = static_padding[0] + half_padding_height
        dynamic_padding[2]    = static_padding[1] + (padding_height - half_padding_height)
        dynamic_padding[1]    = static_padding[2] + half_padding_width
        dynamic_padding[3]    = static_padding[3] + (padding_width - half_padding_width)
    print(input_size, dynamic_padding)
    return dynamic_padding


def construct_tf_pad_op(tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag = False):
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list, node_output_list, input_names, output_names = get_node_input_output(tsm_nodes_info, node_index)
    assert len(node_input_list) == 2
    assert len(node_output_list) == 1
    assert node['format'] == 'NCHW'
    auto_pad = 'NOTSET'
    out_index = node_output_list[0]
    assert node_index in tsm_nodes_info[out_index]['input_index']
    if log_flag:
        print("construct pooling_pad with:")
        print("auto_pad: {}".format(auto_pad))
        print("out_index: {}".format(out_index))
    tsm_nodes_info[out_index]['bubble_info']['auto_pad'] = auto_pad
    tsm_nodes_info[out_index]['bubble_info']['pads'] = compute_tf_pad_value(tsm_nodes_info, node_index, onnx_nodes_info)
    return True


def construct_pooling_v2_op(tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag = False):
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list = node_info['input_index']
    input_names = []
    input_names.append(tsm_nodes_info[node_input_list[0]]['bubble_info']['name'])
    output_names = [node['name'], ]
    assert len(node_input_list) == 4
    if 'type' in node.keys():
        if node['type'].tolist()[0] == 0:
            actual_op_name = 'MaxPool'
        else:
            actual_op_name = 'AveragePool'
    else:
        actual_op_name = 'MaxPool'
    
    if 'pads' not in node.keys() or 'auto_pad' not in node.keys():
        return False
    auto_pad = node['auto_pad']
    pads = node['pads']
    kernel_shape = tsm_nodes_info[node_input_list[2]]['bubble_info']['value'].tolist()[2:]
    strides = tsm_nodes_info[node_input_list[3]]['bubble_info']['value'].tolist()[2:]
    if log_flag:
        print("construct pooling with:")
        print("auto_pad: {}".format(auto_pad))
        print("pads: {}".format(pads))
        print("kernel_shape: {}".format(kernel_shape))
        print("strides: {}".format(strides))
    
    
    infer_shape_dims(onnx_op_name, tsm_nodes_info, node_index)
    node_def = helper.make_node(
        actual_op_name,
        inputs=input_names,
        outputs=output_names,
        auto_pad=auto_pad,
        pads=pads,
        kernel_shape=kernel_shape,
        strides=strides,
    )
    onnx_nodes_info['node'].append(node_def)
    return True

def construct_pooling_op(tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag = False):
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list = node_info['input_index']
    input_names = []
    input_names.append(tsm_nodes_info[node_input_list[0]]['bubble_info']['name'])
    output_names = [node['name'], ]
    assert len(node_input_list) == 1
    assert node['format'] == 'NCHW'
    if 'type' in node.keys():
        if node['type'].tolist()[0] == 0:
            actual_op_name = 'MaxPool'
        else:
            actual_op_name = 'AveragePool'
    else:
        actual_op_name = 'MaxPool'
    
    auto_pad = 'NOTSET'
    pads = node['padding'][2:].transpose((1,0)).reshape((1, -1)).tolist()[0]
    kernel_shape = node['ksize'].tolist()[2:]
    strides = node['stride'].tolist()[2:]
    ceil_mode = 1
    if log_flag:
        print("construct pooling with:")
        print("auto_pad: {}".format(auto_pad))
        print("pads: {}".format(pads))
        print("kernel_shape: {}".format(kernel_shape))
        print("strides: {}".format(strides))
    
    
    infer_shape_dims(onnx_op_name, tsm_nodes_info, node_index)
    node_def = helper.make_node(
        actual_op_name,
        inputs=input_names,
        outputs=output_names,
        auto_pad=auto_pad,
        ceil_mode=ceil_mode,
        pads=pads,
        kernel_shape=kernel_shape,
        strides=strides,
    )
    onnx_nodes_info['node'].append(node_def)
    return True


def construct_concat_op(tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag = False):
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list, node_output_list, input_names, output_names = get_node_input_output(tsm_nodes_info, node_index)
    assert len(node_input_list) >= 1
    assert 'dim' in node.keys()
    assert len(node['dim']) == 1
    axis = node['dim'][0]
    if log_flag:
        print("construct concat with:")
        print("axis: {}".format(axis))
    
    infer_shape_dims(onnx_op_name, tsm_nodes_info, node_index)
    node_def = helper.make_node(
        onnx_op_name,
        inputs=input_names,
        outputs=output_names,
        axis=axis,
    )
    onnx_nodes_info['node'].append(node_def)
    return True


def construct_reshape_op(tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag = False):
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list, node_output_list, input_names, output_names = get_node_input_output(tsm_nodes_info, node_index)
    if len(node_input_list) == 1:
        assert 'shape' in node.keys()
        shape_value = node['shape']
        init_tensor_data = shape_value.astype(np.int64).copy()
        init_tensor_name = node['name'] + "_reshape_value"
        init_tensor_type = conver_numpy_type_to_onnx_type(shape_value.astype(np.int64))
        init_tensor_shape = list(shape_value.shape)
        init_tensor_data = init_tensor_data.reshape(1,-1)[0]
        shape_attr_node_def = helper.make_tensor(init_tensor_name, init_tensor_type, init_tensor_shape, init_tensor_data)
        onnx_nodes_info['initializer'].append(shape_attr_node_def)
        input_names.append(init_tensor_name)
        if log_flag:
            print("construct reshape with:")
            print("tensor name: {}".format(init_tensor_name))
            print("tensor type: {}".format(init_tensor_type))
            print("tensor shape: {}".format(init_tensor_shape))
            print("tensor data: {}".format(init_tensor_data))
    else:
        # reshape op reqiure shape value should be int64
        input_shape_value_index = node_input_list[1]
        shape_node = tsm_nodes_info[input_shape_value_index]['bubble_info']
        if '<const>' == shape_node['op']:
            shape_node['value'].astype(np.int64)
            for init_index in range(len(onnx_nodes_info['initializer'])):
                init_node = onnx_nodes_info['initializer'][init_index]
                if init_node['name'] == shape_node['name']:
                    break
            assert init_index < len(onnx_nodes_info['initializer'])
            del onnx_nodes_info['initializer'][init_index]
            construct_const_op('<const>', 'Const', tsm_nodes_info, input_shape_value_index, onnx_nodes_info, log_flag)
    infer_shape_dims(onnx_op_name, tsm_nodes_info, node_index)
    node_def = helper.make_node(
        onnx_op_name,
        inputs=input_names,
        outputs=output_names,
    )
    onnx_nodes_info['node'].append(node_def)
    return True

def construct_flatten_op(tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag = False):
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list, node_output_list, input_names, output_names = get_node_input_output(tsm_nodes_info, node_index)
    assert len(node_input_list) == 1
    if 'dim' in node.keys():
        assert len(node['dim']) == 1
        axis = node['dim'][0]
    else:
        axis = 1
    if log_flag:
        print("construct flatten with:")
        print("axis: {}".format(axis))

    infer_shape_dims(onnx_op_name, tsm_nodes_info, node_index)
    node_def = helper.make_node(
        onnx_op_name,
        inputs=input_names,
        outputs=output_names,
        axis=axis,
    )
    onnx_nodes_info['node'].append(node_def)
    return True

def construct_transpose_op(tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag = False):
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list, node_output_list, input_names, output_names = get_node_input_output(tsm_nodes_info, node_index)
    assert len(node_input_list) == 1
    assert 'permute' in node.keys()
    perm = node['permute'].tolist()
    if log_flag:
        print("construct transpose with:")
        print("perm: {}".format(perm))
    
    infer_shape_dims(onnx_op_name, tsm_nodes_info, node_index)
    node_def = helper.make_node(
        onnx_op_name,
        inputs=input_names,
        outputs=output_names,
        perm=perm,
    )
    onnx_nodes_info['node'].append(node_def)
    return True


def construct_softmax_op(tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag = False):
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list, node_output_list, input_names, output_names = get_node_input_output(tsm_nodes_info, node_index)
    assert len(node_input_list) == 1
    assert 'dim' in node.keys()
    assert len(node['dim']) == 1
    axis = node['dim'][0]
    if log_flag:
        print("construct softmax with:")
        print("axis: {}".format(axis))
    
    infer_shape_dims(onnx_op_name, tsm_nodes_info, node_index)
    node_def = helper.make_node(
        onnx_op_name,
        inputs=input_names,
        outputs=output_names,
        axis=axis,
    )
    onnx_nodes_info['node'].append(node_def)
    return True


def construct_gather_op(tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag = False):
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list, node_output_list, input_names, output_names = get_node_input_output(tsm_nodes_info, node_index)
    assert len(node_input_list) == 2
    if 'axis' in node.keys():
        assert len(node['axis']) == 1
        axis = node['axis'][0]
    else:
        axis = 0
    if log_flag:
        print("construct gather with:")
        print("axis: {}".format(axis))
    
    infer_shape_dims(onnx_op_name, tsm_nodes_info, node_index)
    node_def = helper.make_node(
        onnx_op_name,
        inputs=input_names,
        outputs=output_names,
        axis=axis,
    )
    onnx_nodes_info['node'].append(node_def)


def construct_clip_op(tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag = False):
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list, node_output_list, input_names, output_names = get_node_input_output(tsm_nodes_info, node_index)
    assert len(node_input_list) == 1
    assert 'max' in node.keys()
    
    min_value = np.float32(0.0)
    min_index = add_const_node_to_tsm(tsm_nodes_info, node_index, min_value, log_flag)
    construct_const_op('<const>', 'Const', tsm_nodes_info, min_index, onnx_nodes_info, log_flag)

    max_value = np.float32(node['max'][0])
    max_index = add_const_node_to_tsm(tsm_nodes_info, node_index, max_value, log_flag)
    construct_const_op('<const>', 'Const', tsm_nodes_info, max_index, onnx_nodes_info, log_flag)

    node_input_list.append(min_index)
    input_names.append(tsm_nodes_info[min_index]['bubble_info']['name'])
    node_input_list.append(max_index)
    input_names.append(tsm_nodes_info[max_index]['bubble_info']['name'])

    if log_flag:
        print("construct clip with:")
    
    infer_shape_dims(onnx_op_name, tsm_nodes_info, node_index)
    node_def = helper.make_node(
        onnx_op_name,
        inputs=input_names,
        outputs=output_names,
    )
    onnx_nodes_info['node'].append(node_def)


def construct_conv2d_op(tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag = False):
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list, node_output_list, input_names, output_names = get_node_input_output(tsm_nodes_info, node_index)
    assert len(node_input_list) >= 2
    # assert node['format'] == "NCHW"
    assert 'dilation' in node.keys()
    dilations = node['dilation'].tolist()[2:]
    assert 'stride' in node.keys()
    strides = node['stride'].tolist()[2:]
    
    if 'conv2d' == tsm_op_name:
        assert 'padding' in node.keys()
        pads = node['padding'].copy()[2:,:].transpose((1,0)).reshape((1,-1))[0].tolist()
    elif 'conv2d_v2' == tsm_op_name:
        assert 'pads' in node.keys()
        assert len(node_input_list) >= 3
        del input_names[1]
        # input_names = []
        # input_names.append(tsm_nodes_info[node_input_list[0]]['bubble_info']['name'])
        # input_names.append(tsm_nodes_info[node_input_list[2]]['bubble_info']['name'])
        pads = node['pads']
    else:
        assert False
    if log_flag:
        print("construct conv2d with:")
        print("strides: {}".format(strides))
        print("dilations: {}".format(dilations))
        print("pads: {}".format(pads))
    
    infer_shape_dims(onnx_op_name, tsm_nodes_info, node_index)
    node_def = onnx.helper.make_node(
        onnx_op_name,
        inputs=input_names,
        outputs=output_names,
        strides=strides,
        dilations=dilations,
        pads=pads,
    )

    onnx_nodes_info['node'].append(node_def)
    return True

def construct_depthwise_conv2d_op(tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag = False):
    onnx_op_name = 'Conv'
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list, node_output_list, input_names, output_names = get_node_input_output(tsm_nodes_info, node_index)
    assert len(node_input_list) >= 2
    assert node['format'] == "NCHW"
    assert 'dilation' in node.keys()
    dilations = node['dilation'].tolist()[2:]
    assert 'stride' in node.keys()
    strides = node['stride'].tolist()[2:]
    if 'depthwise_conv2d' == tsm_op_name:
        assert 'padding' in node.keys()
        pads = node['padding'].copy()[2:,:].transpose((1,0)).reshape((1,-1))[0].tolist()
        weight_node = tsm_nodes_info[node_input_list[1]]['bubble_info']
        group = weight_node['value'].shape[0]
    elif 'depthwise_conv2d_v2' == tsm_op_name:
        assert 'pads' in node.keys()
        assert len(node_input_list) >= 3
        del input_names[1]
        # input_names = []
        # input_names.append(tsm_nodes_info[node_input_list[0]]['bubble_info']['name'])
        # input_names.append(tsm_nodes_info[node_input_list[2]]['bubble_info']['name'])
        pads = node['pads']
        weight_node = tsm_nodes_info[node_input_list[2]]['bubble_info']
        group = weight_node['value'].shape[0]
    else:
        assert False    
    if log_flag:
        print("construct conv2d with:")
        print("strides: {}".format(strides))
        print("dilations: {}".format(dilations))
        print("pads: {}".format(pads))
    
    infer_shape_dims(onnx_op_name, tsm_nodes_info, node_index)
    node_def = onnx.helper.make_node(
        onnx_op_name,
        inputs=input_names,
        outputs=output_names,
        group=group,
        strides=strides,
        dilations=dilations,
        pads=pads,
    )
    onnx_nodes_info['node'].append(node_def)
    return True

def construct_concatfromsequence_op(tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag = False):
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list, node_output_list, input_names, output_names = get_node_input_output(tsm_nodes_info, node_index)
    assert len(node_input_list) >= 1
    assert 'axis' in node.keys()
    assert len(node['axis']) == 1
    axis = node['axis'][0]
    new_axis = 1
    if log_flag:
        print("construct concatfromsequence with:")
        print("axis: {}".format(axis))
        print("new_axis: {}".format(new_axis))
    
    infer_shape_dims(onnx_op_name, tsm_nodes_info, node_index)
    node_def = helper.make_node(
        onnx_op_name,
        inputs=input_names,
        outputs=output_names,
        axis = axis,
        new_axis = new_axis,
    )
    onnx_nodes_info['node'].append(node_def)
    return True


def construct_unsqueeze_op(tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag = False):
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list, node_output_list, input_names, output_names = get_node_input_output(tsm_nodes_info, node_index)
    assert len(node_input_list) == 1
    assert 'axis' in node.keys()
    axes = node['axis'].tolist()
    
    if log_flag:
        print("construct unsqueeze with:")
        print("axes: {}".format(axes))
    
    infer_shape_dims(onnx_op_name, tsm_nodes_info, node_index)
    node_def = helper.make_node(
        onnx_op_name,
        inputs=input_names,
        outputs=output_names,
        axes = axes,
    )
    onnx_nodes_info['node'].append(node_def)
    return True


def construct_batchnormalization_op(tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag = False):
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list, node_output_list, input_names, output_names = get_node_input_output(tsm_nodes_info, node_index)
    assert len(node_input_list) == 5 or len(node_input_list) == 3
    assert 'dim' in node.keys()
    dim = node['dim'].tolist()[0]
    assert dim == 1
    assert 'epsilon' in node.keys()
    epsilon = node['epsilon'].tolist()[0]

    if log_flag:
        print("construct batchnormalization with:")
        print("epsilon: {}".format(epsilon))
    if len(node_input_list) == 3:
        mean_node_info = tsm_nodes_info[node_input_list[1]]
        mean_node = mean_node_info['bubble_info']
        scale_value = np.ones(mean_node['value'].shape).astype(np.float32)
        scale_index = add_const_node_to_tsm(tsm_nodes_info, node_index, scale_value, log_flag)
        construct_const_op('<const>', 'Const', tsm_nodes_info, scale_index, onnx_nodes_info, log_flag)
        node_input_list.insert(1, scale_index)

        bias_value = np.zeros(mean_node['value'].shape).astype(np.float32)
        bias_index = add_const_node_to_tsm(tsm_nodes_info, node_index, bias_value, log_flag)
        construct_const_op('<const>', 'Const', tsm_nodes_info, bias_index, onnx_nodes_info, log_flag)
        node_input_list.insert(2, bias_index)

        node_input_list, node_output_list, input_names, output_names = get_node_input_output(tsm_nodes_info, node_index)

    infer_shape_dims(onnx_op_name, tsm_nodes_info, node_index)
    node_def = helper.make_node(
        onnx_op_name,
        inputs=input_names,
        outputs=output_names,
        epsilon = epsilon,
    )
    onnx_nodes_info['node'].append(node_def)
    return True


def construct_cast_op(tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag = False):
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list, node_output_list, input_names, output_names = get_node_input_output(tsm_nodes_info, node_index)
    assert len(node_input_list) == 1
    if 'to_float' == node['op']:
        to = convert_onnx_type_to_value('FLOAT')
    elif '_cast' == node['op']:
        onnx_type = convert_tsm_type_to_onnx_type(node['dtype'][0])
        to = convert_onnx_type_to_value(onnx_type)
    else:
        print("not supported cast type {}".format(tsm_op_name))
        assert 0
    if log_flag:
        print("construct cast with:")
        print("to: {}".format(to))
    
    infer_shape_dims(onnx_op_name, tsm_nodes_info, node_index)
    node_def = helper.make_node(
        onnx_op_name,
        inputs=input_names,
        outputs=output_names,
        to = to,
    )
    onnx_nodes_info['node'].append(node_def)
    return True



def construct_gemm_op(tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag = False):
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list, node_output_list, input_names, output_names = get_node_input_output(tsm_nodes_info, node_index)
    assert len(node_input_list) >= 2
    if log_flag:
        print("construct gemm with:")
        if 'transpose' in node.keys():
            print("transpose: {}".format(node['transpose']))
        else:
            print("transpose: {}".format(False))

    infer_shape_dims(onnx_op_name, tsm_nodes_info, node_index)
    alpha = 1.0
    beta = 1.0
    transA = 0
    if 'transpose' in node.keys():
        transB = int(node['transpose'])
    else:
        transB = 0
    node_def = helper.make_node(
        onnx_op_name,
        inputs=input_names,
        outputs=output_names,
        alpha=alpha,
        beta=beta,
        transA=transA,
        transB=transB,
    )
    onnx_nodes_info['node'].append(node_def)
    return True


def get_current_graph_output(tsm_nodes_info, node_index, onnx_nodes_info):
    global net_input_shape
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list = node_info['input_index']
    init_tensor_name = tsm_nodes_info[node_input_list[0]]['bubble_info']['name']
    init_tensor_shape = [-1, -1, -1, -1]
    # init_tensor_shape = net_input_shape
    init_tensor_type = TensorProto.FLOAT
    output_tensor = helper.make_tensor_value_info(init_tensor_name, init_tensor_type, init_tensor_shape)
    onnx_initializer_list = onnx_nodes_info['initializer']
    onnx_input_list = onnx_nodes_info['input']
    onnx_output_list = [output_tensor, ]
    onnx_node_list = onnx_nodes_info['node']
    onnx_graph_def = helper.make_graph(onnx_node_list, 'pooling_pad_graph', onnx_input_list, onnx_output_list, initializer=onnx_initializer_list)
    op = onnx.OperatorSetIdProto()
    op.version = 11
    onnx_mode_def = helper.make_model(onnx_graph_def, producer_name='csta_parser', opset_imports=[op])
    onnx_mode_def.ir_version = 6
    onnx.checker.check_model(onnx_mode_def)
    onnx_file_name = "./temp.onnx"
    onnx.save(onnx_mode_def, onnx_file_name)
    onnx_session = onnxruntime.InferenceSession(onnx_file_name)
    input_names = []
    for onnx_node in onnx_session.get_inputs():
        input_names.append(onnx_node.name)
    input_feed = {}
    for item in input_names:
        input_feed[item] = np.zeros(net_input_shape).astype(np.float32)
        # input_feed[item] = np.zeros(init_tensor_shape).astype(np.float32)
    
    pred_result = onnx_session.run([], input_feed=input_feed)
    return pred_result


def add_const_node_to_tsm(tsm_nodes_info, out_index, nump_arr, log_flag = False):
    node_index = len(tsm_nodes_info)
    tsm_op_name = "const_" + str(node_index)
    onnx_op_name = '<const>'
    new_node_info = {}
    new_node_info['bubble_info'] = {}
    new_node_info['input_index'] = []
    new_node_info['output_index'] = []
    new_node_info['output_index'].append(out_index)
    
    new_node_info['bubble_info']['name'] = tsm_op_name
    new_node_info['bubble_info']['op'] = onnx_op_name
    new_node_info['bubble_info']['value'] = nump_arr
    
    init_tensor_name = tsm_op_name
    init_tensor_type = conver_numpy_type_to_onnx_type(nump_arr)
    init_tensor_data = nump_arr
    init_tensor_shape = nump_arr.shape
    if log_flag:
        print("add const tensor with:")
        print("tensor name: {}".format(init_tensor_name))
        print("tensor shape: {}".format(init_tensor_shape))
        print("tensor type: {}".format(init_tensor_type))
        print("tensor data: {}".format(init_tensor_data))

    tsm_nodes_info.append(new_node_info)
    return node_index


def construct_sip_op(tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag = False):
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list, node_output_list, input_names, output_names = get_node_input_output(tsm_nodes_info, node_index)
    assert len(node_input_list) == 2
    infer_result = get_current_graph_output(tsm_nodes_info, node_index, onnx_nodes_info)
    matrixa_shape = infer_result[0].shape
    assert len(matrixa_shape) == 2
    for out_node_index in node_output_list:
        next_node_info = tsm_nodes_info[out_node_index]
        next_node = next_node_info['bubble_info']
        assert next_node['op'] == 'inner_prod'
        next_node_input_list = next_node_info['input_index']
        assert len(next_node_input_list) == 0
        assert next_node_input_list[0] == node_index
        next_node_other_input_info = tsm_nodes_info[next_node_input_list[1]]
        matrixb = next_node_other_input_info['bubble_info']['value']
        matrixb_shape = matrixb.shape
        assert(len(matrixb_shape))
        transpose = next_node_other_input_info['bubble_info']['transpose']
        if transpose:
            assert matrixa_shape[1] > matrixb_shape[1]
            new_matrixb = np.zeros((matrixb_shape[0], matrixa_shape[1]), dtype=matrixb.dtype)
            new_matrixb[:, 0:matrixb_shape[1]] = matrixb
        else:
            assert matrixa_shape[1] > matrixb_shape[0]
            new_matrixb = np.zeros((matrixa_shape[1], matrixb_shape[1]), dtype=matrixb.dtype)
            new_matrixb[0:matrixb_shape[0], :] = matrixb
        new_node_index = add_const_node_to_tsm(tsm_nodes_info, out_node_index, new_matrixb, log_flag)
        construct_const_op("", "", tsm_nodes_info, node_index, onnx_nodes_info, log_flag)
        next_node_input_list[1] = new_node_index
    # remove_node_from_tsm(tsm_nodes_info, node_index)
    # print(infer_result[0].shape)


def construct_i1o1_op(tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag = False):
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list, node_output_list, input_names, output_names = get_node_input_output(tsm_nodes_info, node_index)
    assert len(node_input_list) == 1
    
    infer_shape_dims(onnx_op_name, tsm_nodes_info, node_index)
    node_def = helper.make_node(
        onnx_op_name,
        inputs=input_names,
        outputs=output_names,
    )
    onnx_nodes_info['node'].append(node_def)
    return True


def construct_i2o1_op(tsm_op_name, onnx_op_name, tsm_nodes_info, node_index, onnx_nodes_info, log_flag = False):
    node_info = tsm_nodes_info[node_index]
    node = node_info['bubble_info']
    node_input_list, node_output_list, input_names, output_names = get_node_input_output(tsm_nodes_info, node_index)
    assert len(node_input_list) == 2
    
    infer_shape_dims(onnx_op_name, tsm_nodes_info, node_index)
    node_def = helper.make_node(
        onnx_op_name,
        inputs=input_names,
        outputs=output_names,
    )
    onnx_nodes_info['node'].append(node_def)
    return True


def init_onnx_construct_obj():
    onnx_obj = ConstructOnnxOp()
    onnx_obj.register('_onnx_pooling2d_padding', 'Pooling_pad', construct_pooling_pad_op)
    onnx_obj.register('pooling2d_v2', 'Pooling', construct_pooling_v2_op)
    onnx_obj.register('pooling2d', 'Pooling', construct_pooling_op)
    onnx_obj.register('concat', 'Concat', construct_concat_op)
    onnx_obj.register('_reshape', 'Reshape', construct_reshape_op)
    onnx_obj.register('_reshape_v2', 'Reshape', construct_reshape_op)
    onnx_obj.register('flatten', 'Flatten', construct_flatten_op)
    onnx_obj.register('_transpose', 'Transpose', construct_transpose_op)
    onnx_obj.register('softmax', 'Softmax', construct_softmax_op)
    onnx_obj.register('conv2d', 'Conv', construct_conv2d_op)
    onnx_obj.register('conv2d_v2', 'Conv', construct_conv2d_op)
    onnx_obj.register('depthwise_conv2d', 'DepthWiseConv', construct_depthwise_conv2d_op)
    onnx_obj.register('_tf_conv2d_padding', 'tf_conv2d_pad', construct_tf_pad_op)
    onnx_obj.register('depthwise_conv2d_v2', 'DepthWiseConv', construct_depthwise_conv2d_op)
    onnx_obj.register('gather', 'Gather', construct_gather_op)
    onnx_obj.register('relu_max', 'Clip', construct_clip_op)
    # onnx_obj.register('stack', 'ConcatFromSequence', construct_concatfromsequence_op)
    onnx_obj.register('unsqueeze', 'Unsqueeze', construct_unsqueeze_op)
    onnx_obj.register('to_float', 'Cast', construct_cast_op)
    onnx_obj.register('_cast', 'Cast', construct_cast_op)
    onnx_obj.register('inner_prod', 'Gemm', construct_gemm_op)
    onnx_obj.register('batch_norm', 'BatchNormalization', construct_batchnormalization_op)
    # onnx_obj.register('shape_index_patch', 'ShapeIndexPatch', construct_sip_op) # specical case, next node must be inner_prod
    onnx_obj.register('add', 'Add', construct_i2o1_op)
    onnx_obj.register('sub', 'Sub', construct_i2o1_op)
    onnx_obj.register('mul', 'Mul', construct_i2o1_op)
    onnx_obj.register('sigmoid', 'Sigmoid', construct_i1o1_op)
    onnx_obj.register('relu', 'Relu', construct_i1o1_op)
    onnx_obj.register('_shape', 'Shape', construct_i1o1_op)
    onnx_obj.register('<const>', 'Const', construct_const_op)
    onnx_obj.register('input', 'Input', construct_input_op)
    onnx_obj.register('output', 'Output', construct_output_op)
    return onnx_obj


def merge_conv_ip_addbias_op(tsm_nodes_info, log_verbose):
    for index in range(len(tsm_nodes_info)):
        node_info = tsm_nodes_info[index]
        op_name = node_info['bubble_info']['op']
        node_name = node_info['bubble_info']['name']
        if op_name == 'conv2d' or op_name == 'inner_prod' or op_name == "depthwise_conv2d" or \
            op_name == 'conv2d_v2' or op_name == 'depthwise_conv2d_v2':
            node_input_list = node_info['input_index']
            node_output_list = node_info['output_index']
            for out_index in node_output_list:
                next_node_info = tsm_nodes_info[out_index]
                next_node_input_list = next_node_info['input_index']
                next_node_output_list = next_node_info['output_index']
                next_op_name = next_node_info['bubble_info']['op']
                next_node_name = next_node_info['bubble_info']['name']
                if next_op_name == "add_bias":
                    assert len(next_node_input_list) == 2
                    next_node_info['old_input_index'] = copy.deepcopy(next_node_input_list)
                    next_node_info['old_output_index'] = copy.deepcopy(next_node_output_list)
                    # replace next next node input 
                    for item in next_node_output_list:
                        next_next_node_info = tsm_nodes_info[item]
                        next_next_input_list = next_next_node_info['input_index']
                        for next_next_index in range(len(next_next_input_list)):
                            if next_next_input_list[next_next_index] == out_index:
                                next_next_input_list[next_next_index] = index
                    # add conv node input with bias
                    node_input_list.append(next_node_input_list[1])
                    # remove bias from conv out
                    node_output_list.remove(out_index)
                    # add bias out to conv out
                    node_output_list.extend(copy.deepcopy(next_node_output_list))
                    next_node_info['input_index'] = []
                    next_node_info['output_index'] = []
                    if log_verbose:
                        print("merge {}:{} {}:{} and {}:{}".format(index, out_index, node_name, op_name, next_node_name, next_op_name))
                    break
            if op_name == "depthwise_conv2d":
                node_input_list = node_info['input_index']
                weight_node = tsm_nodes_info[node_input_list[1]]['bubble_info']
                weight_node['value'] = weight_node['value'].transpose((1, 0, 2, 3))
            if op_name == "depthwise_conv2d_v2":
                node_input_list = node_info['input_index']
                weight_node = tsm_nodes_info[node_input_list[2]]['bubble_info']
                weight_node['value'] = weight_node['value'].transpose((1, 0, 2, 3))


def merge_batch_norm_batch_scale_op(tsm_nodes_info, log_verbose):
    for index in range(len(tsm_nodes_info)):
        node_info = tsm_nodes_info[index]
        op_name = node_info['bubble_info']['op']
        node_name = node_info['bubble_info']['name']
        if op_name == 'batch_norm':
            node_input_list = node_info['input_index']
            node_output_list = node_info['output_index']
            for out_index in node_output_list:
                next_node_info = tsm_nodes_info[out_index]
                next_node_input_list = next_node_info['input_index']
                next_node_output_list = next_node_info['output_index']
                next_op_name = next_node_info['bubble_info']['op']
                next_node_name = next_node_info['bubble_info']['name']
                if next_op_name == "batch_scale":
                    assert len(next_node_input_list) == 3
                    next_node_info['old_input_index'] = copy.deepcopy(next_node_input_list)
                    next_node_info['old_output_index'] = copy.deepcopy(next_node_output_list)
                    # replace next next node input 
                    for item in next_node_output_list:
                        next_next_node_info = tsm_nodes_info[item]
                        next_next_input_list = next_next_node_info['input_index']
                        for next_next_index in range(len(next_next_input_list)):
                            if next_next_input_list[next_next_index] == out_index:
                                next_next_input_list[next_next_index] = index
                    # add conv node input with scale and bias
                    node_input_list.insert(1, next_node_input_list[1])
                    node_input_list.insert(2, next_node_input_list[2])
                    # remove batch scale from batch norm out
                    node_output_list.remove(out_index)
                    # add bias out to conv out
                    node_output_list.extend(copy.deepcopy(next_node_output_list))
                    next_node_info['input_index'] = []
                    next_node_info['output_index'] = []
                    if log_verbose:
                        print("merge {}:{} {}:{} and {}:{}".format(index, out_index, node_name, op_name, next_node_name, next_op_name))
                    break

def merge_ops(tsm_nodes_info, log_verbose):
    merge_conv_ip_addbias_op(tsm_nodes_info, log_verbose)
    merge_batch_norm_batch_scale_op(tsm_nodes_info, log_verbose)

def replace_stack_op(tsm_nodes_info, log_verbose):
    node_count = len(tsm_nodes_info)
    insert_new_nodes = []
    for index in range(len(tsm_nodes_info)):
        node_info = tsm_nodes_info[index]
        node = node_info['bubble_info']
        op_name = node_info['bubble_info']['op']
        node_name = node_info['bubble_info']['name']
        if op_name == 'stack':
            insert_unsqueeze_nodes_index = []
            node_input_list = node_info['input_index']
            node_output_list = node_info['output_index']
            assert len(node_input_list) > 1
            assert 'axis' in node.keys()
            if log_verbose:
                print("{}: replace {} with unsqueeze and concat node".format(index, node_name))
            # add new unsqueeze node
            for item_index in range(len(node_input_list)):
                node_index = node_input_list[item_index]
                new_unsqueeze_node_index = node_count + item_index
                new_unsqueeze_node = {}
                new_unsqueeze_node['bubble_info'] = {}
                new_unsqueeze_node['bubble_info']['op'] = 'unsqueeze'
                new_unsqueeze_node['bubble_info']['name'] = node_name + '_unsqueeze'
                new_unsqueeze_node['bubble_info']['axis'] = node['axis'].copy()
                new_unsqueeze_node['input_index'] = [node_index, ]
                new_unsqueeze_node['output_index'] = [node_count + len(node_input_list), ]
                insert_unsqueeze_nodes_index.append(node_count + item_index)
                insert_new_nodes.append(new_unsqueeze_node)
                # update origin pre node output
                output_list = tsm_nodes_info[node_index]['output_index']
                assert index in output_list
                for _index in range(len(output_list)):
                    item = output_list[_index]
                    if item == index:
                        output_list[_index] = new_unsqueeze_node_index
            
            # add new unsqueeze node
            new_concat_node = {}
            new_concat_node['bubble_info'] = {}
            new_concat_node['bubble_info']['op'] = 'concat'
            new_concat_node['bubble_info']['name'] = node_name + '_concat'
            new_concat_node['bubble_info']['dim'] = node['axis'].copy()
            new_concat_node['input_index'] = copy.deepcopy(insert_unsqueeze_nodes_index)
            new_concat_node['output_index'] = copy.deepcopy(node_output_list)
            insert_new_nodes.append(new_concat_node)

            # update origin next node input
            for node_index in node_output_list:
                input_list = tsm_nodes_info[node_index]['input_index']
                assert index in input_list
                for item_index in range(len(input_list)):
                    item = input_list[item_index]
                    if item == index:
                        input_list[item_index] = node_count + len(node_input_list)
            
            # clear and backup origin node input output
            node_count = node_count + len(insert_new_nodes)
            node_info['old_input_index'] = copy.deepcopy(node_info['input_index'])
            node_info['old_output_index'] = copy.deepcopy(node_info['output_index'])
            node_info['input_index'] = []
            node_info['output_index'] = []
    tsm_nodes_info.extend(insert_new_nodes)


def replace_unsupported_ops(tsm_nodes_info, log_verbose):
    replace_stack_op(tsm_nodes_info, log_verbose)


def remove_unspported_ops(tsm_nodes_info, log_verbose):
    for index in range(len(tsm_nodes_info)):
        remove_flag = False
        node_info = tsm_nodes_info[index]
        op_name = node_info['bubble_info']['op']
        node_name = node_info['bubble_info']['name']
        if op_name == '_limit' or op_name == '_copy' or op_name == '_dimshuffle' or \
            op_name == '_resize2d' or op_name == 'crop_nd':
            node_input_list = node_info['input_index']
            assert len(node_input_list) >= 1
            node_output_list = node_info['output_index']
            pre_node_info = tsm_nodes_info[node_input_list[0]]
            # remove current from pre node output
            pre_node_info['output_index'].remove(index)
            # insert next to pre node output
            for item in node_output_list:
                if item not in pre_node_info['output_index']:
                    pre_node_info['output_index'].append(item)
            
            for node_index in node_output_list:
                next_node_info = tsm_nodes_info[node_index]
                next_node_input_list = next_node_info['input_index']
                # remove current form next node input 
                # insert pre index to next node input
                for item_index in range(len(next_node_input_list)):
                    if index == next_node_input_list[item_index]:
                        next_node_input_list[item_index] = node_input_list[0]

            # backup input/output index
            node_info['old_input_index'] = copy.deepcopy(node_info['input_index'])
            node_info['old_output_index'] = copy.deepcopy(node_info['output_index'])
            node_info['input_index'] = []
            node_info['output_index'] = []
            remove_flag = True
            if len(node_input_list) == 2:
                pre_node_info = tsm_nodes_info[node_input_list[1]]
                pre_node_info['output_index'].remove(index)

        if remove_flag and log_verbose:
            print("remove node {}: {}".format(index, node_name))

def unique_nodes_name(tsm_nodes_info):
    for index in range(len(tsm_nodes_info)):
        node_info = tsm_nodes_info[index]
        node_name = node_info['bubble_info']['name']
        node_info['bubble_info']['name'] = node_name + "_{}".format(index)


def construct_onnx_from_tsm(tsm_module_info, input_shape, log_verbose=False):
    global net_input_shape
    flag = True
    tsm_nodes_info = tsm_module_info['graph']['nodes_info']
    net_input_shape = input_shape
    remove_unspported_ops(tsm_nodes_info, log_verbose)
    merge_ops(tsm_nodes_info, log_verbose)
    replace_unsupported_ops(tsm_nodes_info, log_verbose)
    unique_nodes_name(tsm_nodes_info)
    onnx_nodes_info = {}
    onnx_nodes_info['initializer'] = []
    onnx_nodes_info['input'] = []
    onnx_nodes_info['output'] = []
    onnx_nodes_info['node'] = []
    init_list = []
    init_name_dict = {}
    flag = get_initializer(tsm_nodes_info, init_list, init_name_dict)
    construct_onnx_obj = init_onnx_construct_obj()
    for key in init_name_dict.keys():
        index = init_name_dict[key]
        flag = construct_onnx_obj.construct_node(tsm_nodes_info, index, onnx_nodes_info, log_verbose)
        if flag == False:
            print("convert {} const node fail, node name is {}".format(index, tsm_nodes_info[index]['bubble_info']['name']))
            break
        tsm_nodes_info[index]['done'] = True

    if flag:
        input_list, out_list, topo_order_list = get_topo_order(tsm_nodes_info)
        # construct input tensors
        for index in input_list:
            flag = construct_onnx_obj.construct_node(tsm_nodes_info, index, onnx_nodes_info, log_verbose)
            if flag == False:
                print("convert {} node fail, node name is {}".format(index, tsm_nodes_info[index]['bubble_info']['name']))
                break
            tsm_nodes_info[index]['done'] = True
    if flag:
        # consturct node 
        for index in topo_order_list:
            flag = construct_onnx_obj.construct_node(tsm_nodes_info, index, onnx_nodes_info, log_verbose)
            if flag == False:
                print("convert {} node fail, node name is {}".format(index, tsm_nodes_info[index]['bubble_info']['name']))
                break
            tsm_nodes_info[index]['done'] = True
    if flag:
        # construct output node 
        for index in out_list:
            flag = construct_onnx_obj.construct_node(tsm_nodes_info, index, onnx_nodes_info, log_verbose)
            if flag == False:
                print("convert {} node fail, node name is {}".format(index, tsm_nodes_info[index]['bubble_info']['name']))
                break
            tsm_nodes_info[index]['done'] = True
    if flag:
        time_stamp = time.localtime()
        doc_string = "created on {}".format(time.strftime('%Y-%m-%d %H-%M-%S', time_stamp))
        onnx_initializer_list = onnx_nodes_info['initializer']
        onnx_input_list = onnx_nodes_info['input']
        onnx_output_list = onnx_nodes_info['output']
        onnx_node_list = onnx_nodes_info['node']
        onnx_graph_def = helper.make_graph(onnx_node_list, 'tsm_graph', onnx_input_list, onnx_output_list, initializer=onnx_initializer_list, doc_string=doc_string)
        op = onnx.OperatorSetIdProto()
        op.version = 11        
        onnx_mode_def = helper.make_model(onnx_graph_def, producer_name='csta_parser', opset_imports=[op])
        onnx_mode_def.ir_version = 6
        checker.check_model(onnx_mode_def)
        helper.printable_graph(onnx_mode_def.graph)
    else:
        onnx_mode_def = None
    return onnx_mode_def, flag

def check_unspported_tsm_op(tsm_module_info):
    construct_onnx_obj = init_onnx_construct_obj()
    tsm_nodes_info = tsm_module_info['graph']['nodes_info']
    support_flag = True
    for index in range(len(tsm_nodes_info)):
        node_info = tsm_nodes_info[index]
        op_name = node_info['bubble_info']['op']
        if op_name not in construct_onnx_obj.tsm_op_onnx_op_map.keys():
            print("index {} op {} not supported.".format(index, op_name))
            support_flag = False
        if op_name == "relu_max":
            print("{} ".format(node_info['bubble_info']['max']))
    
    return support_flag