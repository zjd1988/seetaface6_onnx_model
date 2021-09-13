import os
import struct 
import numpy as np


def bin_data_get_mask(bin_data, offset):
    mask_str_len = 4
    return bin_data[offset:offset+mask_str_len].decode(encoding='UTF-8',errors='strict'), offset + mask_str_len

def bin_data_get_char(bin_data, offset):
    return int.from_bytes(bin_data[offset: offset+1], byteorder='little', signed=False), offset+1

def bin_data_get_int(bin_data, offset):
    return int.from_bytes(bin_data[offset: offset+4], byteorder='little', signed=False), offset+4

def bin_data_get_float(bin_data, offset):
    return struct.unpack("f", bin_data[offset: offset+4])[0], offset+4

def bin_data_get_string(bin_data, offset):
    str_len, offset = bin_data_get_int(bin_data, offset)
    return bin_data[offset:offset+str_len].decode(encoding='UTF-8',errors='strict'), offset + str_len

def parse_type(model_bin_datas, offset):
    type_int, offset = bin_data_get_char(model_bin_datas, offset)
    if type_int == 0:
        return 'NIL', offset
    elif type_int == 1:
        return 'INT', offset 
    elif type_int == 2:
        return 'FLOAT', offset 
    elif type_int == 3:
        return 'STRING', offset 
    elif type_int == 4:
        return 'BINARY', offset 
    elif type_int == 5:
        return 'LIST', offset 
    elif type_int == 6:
        return 'DICT', offset 
    elif type_int == 7:
        return 'BOOLEAN', offset 
    else:
        "print unspported type {}".format(type_int)
        return '', offset 

def parse_nil_data(model_bin_datas, offset):
    nil_data, offset = bin_data_get_char(model_bin_datas, offset)
    return nil_data, offset

def parse_int_data(model_bin_datas, offset):
    int_val, offset = bin_data_get_int(model_bin_datas, offset)
    return int_val, offset

def parse_float_data(model_bin_datas, offset):
    float_val, offset = bin_data_get_float(model_bin_datas, offset)
    return float_val, offset

def parse_string_data(model_bin_datas, offset):
    str_val, offset = bin_data_get_string(model_bin_datas, offset)
    return str_val, offset

def parse_binary_data(model_bin_datas, offset):
    bin_size, offset = bin_data_get_int(model_bin_datas, offset)
    bin_data, offset = model_bin_datas[offset:offset+bin_size], offset+bin_size
    return bin_data, offset

def parse_list_data(model_bin_datas, offset):
    list_size, offset = bin_data_get_int(model_bin_datas, offset)
    list_info = []
    for index in range(list_size):
        data_info, offset = parse_bin_data(model_bin_datas, offset)
        list_info.append(data_info)
    return list_info, offset

def parse_dict_data(model_bin_datas, offset):
    dict_size, offset = bin_data_get_int(model_bin_datas, offset)
    dict_info = {}
    for index in range(dict_size):
        key_str, offset = bin_data_get_string(model_bin_datas, offset)
        data_info, offset = parse_bin_data(model_bin_datas, offset)
        dict_info[key_str] = data_info
    return dict_info, offset

def parse_boolean_data(model_bin_datas, offset):
    bool_data, offset = bin_data_get_char(model_bin_datas, offset)
    return bool(bool_data), offset

def parse_bin_data(model_bin_datas, offset):
    data_type, offset = parse_type(model_bin_datas, offset)
    data_info, offset = parse_data_with_type(model_bin_datas, offset, data_type)
    return data_info, offset

def parse_data_with_type(model_bin_datas, offset, data_type):
    data_info = None
    if data_type == "NIL":
        data_info, offset = parse_nil_data(model_bin_datas, offset)
    elif data_type == "INT":
        data_info, offset = parse_int_data(model_bin_datas, offset)
    elif data_type == "FLOAT":
        data_info, offset = parse_float_data(model_bin_datas, offset)
    elif data_type == "STRING":
        data_info, offset = parse_string_data(model_bin_datas, offset)
    elif data_type == "BINARY":
        data_info, offset = parse_binary_data(model_bin_datas, offset)
    elif data_type == "LIST":
        data_info, offset = parse_list_data(model_bin_datas, offset)
    elif data_type == "DICT":
        data_info, offset = parse_dict_data(model_bin_datas, offset)
    else:
        data_info, offset = parse_boolean_data(model_bin_datas, offset)
    
    return data_info, offset


def parse_csta_bin_data(model_bin_datas):
    offset = 4
    model_info, offset = parse_bin_data(model_bin_datas, offset)
    if offset <= len(model_bin_datas):
        return model_info, True
    else:
        return {}, False

def model_bin_decrypt(model_bytes, key_str):
    cskey = 0
    key = key_str.encode('utf-8')
    for i in range(len(key)):
        cskey *= 10
        cskey += key[i]
    AES_BLOCKLEN = 16
    nremain = len(model_bytes) % AES_BLOCKLEN
    n_valid = len(model_bytes) - nremain
    cs_key_bytes = cskey.to_bytes(length=8, byteorder='little', signed=False)

    # decrypt_model_bytes = copy.deepcopy(model_bytes)
    decrypt_model_bytearray = bytearray(n_valid)
    for i in range(n_valid):
        key_byte = cs_key_bytes[i % 8]
        decrypt_model_bytearray[i] = model_bytes[i] ^ key_byte
    
    return decrypt_model_bytearray


def parse_tsm_module_input(tsm_module_data, offset):
    input_list = []
    list_size, offset = bin_data_get_int(tsm_module_data, offset)
    for i in range(list_size):
        item_value, offset = bin_data_get_int(tsm_module_data, offset)
        input_list.append(item_value)
    return input_list, offset

def parse_tsm_module_output(tsm_module_data, offset):
    output_list = []
    list_size, offset = bin_data_get_int(tsm_module_data, offset)
    for i in range(list_size):
        item_value, offset = bin_data_get_int(tsm_module_data, offset)
        output_list.append(item_value)
    return output_list, offset


# enum DTYPE {
#     VOID        = 0,
#     INT8        = 1,
#     UINT8       = 2,
#     INT16       = 3,
#     UINT16      = 4,
#     INT32       = 5,
#     UINT32      = 6,
#     INT64       = 7,
#     UINT64      = 8,
#     FLOAT16     = 9,
#     FLOAT32     = 10,
#     FLOAT64     = 11,
#     PTR         = 12,              ///< for ptr type, with length of sizeof(void*) bytes
#     CHAR8       = 13,            ///< for char saving string
#     CHAR16      = 14,           ///< for char saving utf-16 string
#     CHAR32      = 15,           ///< for char saving utf-32 string
#     UNKNOWN8    = 16,        ///< for self define type, with length of 1 byte
#     UNKNOWN16   = 17,
#     UNKNOWN32   = 18,
#     UNKNOWN64   = 19,
#     UNKNOWN128  = 20,

#     BOOLEAN     = 21,    // bool type, using byte in native
#     COMPLEX32   = 22,  // complex 32(16 + 16)
#     COMPLEX64   = 23,  // complex 64(32 + 32)
#     COMPLEX128  = 24,  // complex 128(64 + 64)

#     SINK8Q0     = 25,
#     SINK8Q1     = 26,
#     SINK8Q2     = 27,
#     SINK8Q3     = 28,
#     SINK8Q4     = 29,
#     SINK8Q5     = 30,
#     SINK8Q6     = 31,
#     SINK8Q7     = 32,
# };

def get_value_bytes(shape, datatype):
    count = 1
    if len(shape) != 0:
        for item in shape:
            count *= item
    else:
        count = 1
    item_size = 1
    if datatype == 1 or datatype == 2 or datatype == 13 or datatype == 21:
        item_size = 1
    elif datatype == 3 or datatype == 4:
        item_size = 2
    elif datatype == 5 or datatype == 6 or datatype == 10:
        item_size = 4
    elif datatype == 7 or datatype == 8 or datatype == 11:
        item_size = 8
    else:
        print("unspported data type {}".format(datatype))
        return -1
    return count * item_size


def parse_param_item(tsm_module_data, offset):
    item_info = None
    item_shape = []
    # read tensor dtype
    datatype, offset = bin_data_get_char(tsm_module_data, offset)
    if datatype < 0:
        print("wrong data type {}".format(datatype))
        return item_info, offset
    # read tensor shape
    val_shape_size, offset = bin_data_get_int(tsm_module_data, offset)
    for i in range(val_shape_size):
        shape_value, offset = bin_data_get_int(tsm_module_data, offset)
        item_shape.append(shape_value)
    # read tensor data
    tensor_data_size = get_value_bytes(item_shape, datatype)
    if tensor_data_size != -1:
        if datatype == 1:
            item_info = np.frombuffer(tsm_module_data[offset:offset+tensor_data_size], dtype=np.int8)
            if len(item_shape) > 0:
                item_info = item_info.reshape(item_shape)
            offset += tensor_data_size
        elif datatype == 2:
            item_info = np.frombuffer(tsm_module_data[offset:offset+tensor_data_size], dtype=np.uint8)
            if len(item_shape) > 0:
                item_info = item_info.reshape(item_shape)
            offset += tensor_data_size
        elif datatype == 3:
            item_info = np.frombuffer(tsm_module_data[offset:offset+tensor_data_size], dtype=np.int16)
            if len(item_shape) > 0:
                item_info = item_info.reshape(item_shape)
            offset += tensor_data_size
        elif datatype == 4:
            item_info = np.frombuffer(tsm_module_data[offset:offset+tensor_data_size], dtype=np.uint16)
            if len(item_shape) > 0:
                item_info = item_info.reshape(item_shape)
            offset += tensor_data_size
        elif datatype == 5:
            item_info = np.frombuffer(tsm_module_data[offset:offset+tensor_data_size], dtype=np.int32)
            if len(item_shape) > 0:
                item_info = item_info.reshape(item_shape)
            offset += tensor_data_size
        elif datatype == 6:
            item_info = np.frombuffer(tsm_module_data[offset:offset+tensor_data_size], dtype=np.uint32)
            if len(item_shape) > 0:
                item_info = item_info.reshape(item_shape)
            offset += tensor_data_size
        elif datatype == 7:
            item_info = np.frombuffer(tsm_module_data[offset:offset+tensor_data_size], dtype=np.int64)
            if len(item_shape) > 0:
                item_info = item_info.reshape(item_shape)
            offset += tensor_data_size
        elif datatype == 8:
            item_info = np.frombuffer(tsm_module_data[offset:offset+tensor_data_size], dtype=np.uint64)
            if len(item_shape) > 0:
                item_info = item_info.reshape(item_shape)
            offset += tensor_data_size
        elif datatype == 10:
            item_info = np.frombuffer(tsm_module_data[offset:offset+tensor_data_size], dtype=np.float32)
            if len(item_shape) > 0:
                item_info = item_info.reshape(item_shape)
            offset += tensor_data_size
        elif datatype == 11:
            item_info = np.frombuffer(tsm_module_data[offset:offset+tensor_data_size], dtype=np.float64)
            if len(item_shape) > 0:
                item_info = item_info.reshape(item_shape)
            offset += tensor_data_size
        elif datatype == 13:
            item_info = tsm_module_data[offset:offset+tensor_data_size].decode(encoding='UTF-8',errors='strict')
            offset += tensor_data_size
        elif datatype == 21:
            bool_data, offset = bin_data_get_char(tsm_module_data, offset)
            item_info = bool(bool_data)
        else:
            print("unspported data type {}".format(datatype))
    else:
        print("get tensor bytes size error")
    return item_info, offset


def parse_param_info(tsm_module_data, offset):
    param_item_info = []
    # read val list size
    item_list_size, offset = bin_data_get_int(tsm_module_data, offset)
    for i in range(item_list_size):
        item_value, offset = parse_param_item(tsm_module_data, offset)
        param_item_info.append(item_value)
    return param_item_info, offset


def parse_param(tsm_module_data, offset):
    param_info = {}
    # read param name
    param_name, offset = bin_data_get_string(tsm_module_data, offset)
    # read param value
    # if param_name == '#op' or param_name == '#name' or param_name == 'value':
    #     param_val, offset = parse_param_info(tsm_module_data, offset)
    # else:
    #     print("current not support {}".format(param_name))
    param_val, offset = parse_param_info(tsm_module_data, offset)
    param_info[param_name] = param_val
    return param_info, offset


def parse_tsm_module_bubble(tsm_module_data, offset):
    bubble_info = {}
    param_info_dict = {}
    param_size, offset = bin_data_get_int(tsm_module_data, offset)
    for i in range(param_size):
        param_info, offset = parse_param(tsm_module_data, offset)
        param_info_dict = {**param_info_dict, **param_info}
    
    for key in param_info_dict.keys():
        if key == "#op":
            bubble_info["op"] = param_info_dict["#op"][0]
        elif key == "#name":
            bubble_info["name"] = param_info_dict["#name"][0]
        elif key == "#output_count":
            bubble_info["output_count"] = param_info_dict["#output_count"][0]
        else:
            bubble_info[key] = param_info_dict[key][0]

    return bubble_info, offset


def link_node(nodes_info):
    for index in range(len(nodes_info)):
        node = nodes_info[index]
        node_input_list = node['input_index']
        for item in node_input_list:
            if index not in nodes_info[item]['output_index']:
                nodes_info[item]['output_index'].append(index)


def parse_tsm_module_graph(tsm_module_data, offset):
    tsm_graph = {}
    # read node size
    node_size, offset = bin_data_get_int(tsm_module_data, offset)
    nodes_info = []
    for i in range(node_size):
        # .1 read bubble
        bubble_info, offset = parse_tsm_module_bubble(tsm_module_data, offset)
        # .2 read node input index
        node_input_index_list = []
        list_size, offset = bin_data_get_int(tsm_module_data, offset)
        for i in range(list_size):
            item_value, offset = bin_data_get_int(tsm_module_data, offset)
            node_input_index_list.append(item_value)
        node_item = {}
        node_item['bubble_info'] = bubble_info
        node_item['input_index'] = node_input_index_list
        node_item['output_index'] = []
        nodes_info.append(node_item)
    link_node(nodes_info)
    tsm_graph['nodes_info'] = nodes_info
    return tsm_graph, offset

def parse_tsm_module(tsm_module_data):
    tsm_module_info = {}
    offset = 0
    # parse header
    fake, offset = bin_data_get_int(tsm_module_data, offset)
    code, offset = bin_data_get_int(tsm_module_data, offset)
    header_data = tsm_module_data[offset:offset+120]
    offset += 120
    if code != 0x19910929:
        return {}, False
    # parse input
    input_list, offset = parse_tsm_module_input(tsm_module_data, offset)
    tsm_module_info["input"] = input_list
    # parse output
    output_list, offset = parse_tsm_module_output(tsm_module_data, offset)
    tsm_module_info["output"] = output_list
    # parse graph
    tsm_graph, offset = parse_tsm_module_graph(tsm_module_data, offset)
    tsm_module_info["graph"] = tsm_graph

    return tsm_module_info, True

def parse_csta_model_file(csta_model_file):
    csta_model_info = {}
    tsm_module_info = {}
    parse_flag = True
    if os.path.exists(csta_model_file):
        with open(csta_model_file, "rb") as f:
            model_bytes = f.read()
        if len(model_bytes) <= 0:
            print("model file len is zero")
            parse_flag = False
        else:
            offset = 0
            model_mask, offset = bin_data_get_int(model_bytes, offset)
            print("model mask info is {}".format(model_mask))
            if model_mask == 0x74736166:
                key_str = 'seetatech.com'
                decrypt_model_bytes = model_bin_decrypt(model_bytes[offset:], key_str)
                csta_model_info, parse_flag = parse_csta_bin_data(decrypt_model_bytes)
            else:
                parse_flag = False
    else:
        print("{} not exists".format(csta_model_file))
        csta_model_info = {}
        parse_flag = False
    if parse_flag:
        tsm_module_data = csta_model_info['backbone']['tsm']
        tsm_module_info, parse_flag = parse_tsm_module(tsm_module_data)
        if parse_flag == False:
            print("parse tsm module fail")
    return csta_model_info, tsm_module_info, parse_flag