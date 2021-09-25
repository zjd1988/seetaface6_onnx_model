# -*- coding: utf-8 -*-
import cv2
import math
import numpy as np
from common.face_anti_spoofing_anchors import fas_anchors


def decode_box(anchor, box_encoding):
    anchor = [anchor[1], anchor[0], anchor[3], anchor[2]]
    box_encoding = [box_encoding[1], box_encoding[0], box_encoding[3], box_encoding[2]]
    width = anchor[2] - anchor[0]
    height = anchor[3] - anchor[1]
    ctr_x = anchor[0] + 0.5 * width
    ctr_y = anchor[1] + 0.5 * height

    pred_ctr_x = box_encoding[0] * 0.1 * width + ctr_x
    pred_ctr_y = box_encoding[1] * 0.1 * height + ctr_y
    pred_w = math.exp(box_encoding[2] * 0.2) * width
    pred_h = math.exp(box_encoding[3] * 0.2) * height

    region = [pred_ctr_x - 0.5 * pred_w, pred_ctr_y - 0.5 * pred_h, pred_ctr_x + 0.5 * pred_w, pred_ctr_y + 0.5 * pred_h]
    return region


def re_blur(gray_data):
    height = gray_data.shape[0]
    width = gray_data.shape[1]
    blur_val = 0.0
    kernel = [1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0]
    BVer = np.zeros((height, width))
    BHor = np.zeros((height, width))

    filter_data = 0.0
    for i in range(height):
        for j in range(width):
            if i < 4 or i > height - 5:
                BVer[i, j] = gray_data[i, j]
            else:
                filter_data = kernel[0] * gray_data[(i - 4), j] + kernel[1] * gray_data[(i - 3), j] + kernel[2] * gray_data[(i - 2), j] +\
                              kernel[3] * gray_data[(i - 1), j] + kernel[4] * gray_data[i, j] +       kernel[5] * gray_data[(i + 1), j] +\
                              kernel[6] * gray_data[(i + 2), j] + kernel[7] * gray_data[(i + 3), j] + kernel[8] * gray_data[(i + 4), j]
                BVer[i, j] = filter_data

            if j < 4 or j > width - 5:
                BHor[i, j] = gray_data[i, j]
            else:
                filter_data = kernel[0] * gray_data[i, (j - 4)] + kernel[1] * gray_data[i, (j - 3)] + kernel[2] * gray_data[i, (j - 2)] +\
                              kernel[3] * gray_data[i, (j - 1)] + kernel[4] * gray_data[i, j] +       kernel[5] * gray_data[i, (j + 1)] +\
                              kernel[6] * gray_data[i, (j + 2)] + kernel[7] * gray_data[i, (j + 3)] + kernel[8] * gray_data[i, (j + 4)]
                BHor[i, j] = filter_data

    D_Fver = 0.0
    D_FHor = 0.0
    D_BVer = 0.0
    D_BHor = 0.0
    s_FVer = 0.0
    s_FHor = 0.0
    s_Vver = 0.0
    s_VHor = 0.0
    for i in range(height):
        if i == 0:
            continue
        for j in range(width):
            if j == 0:
                continue
            D_Fver = float(abs(gray_data[i, j] - gray_data[(i - 1), j]))
            s_FVer += D_Fver
            D_BVer = float(abs(BVer[i, j] - BVer[(i - 1), j]))
            s_Vver += max(0.0, D_Fver - D_BVer)

            D_FHor = float(abs(gray_data[i, j] - gray_data[i, (j - 1)]))
            s_FHor += D_FHor
            D_BHor = float(abs(BHor[i, j] - BHor[i, (j - 1)]))
            s_VHor += max(0.0, D_FHor - D_BHor)

    b_FVer = (s_FVer - s_Vver) / s_FVer
    b_FHor = (s_FHor - s_VHor) / s_FHor
    blur_val = max( b_FVer, b_FHor )

    return blur_val;    


def clarity_estimate(face):
    if face.shape[0] < 9 or face.shape[1] < 9:
        return 0.0
    gray_data = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur_val = re_blur(gray_data)
    clarity = 1.0 - blur_val

    T1 = 0.3
    T2 = 0.55
    if clarity <= T1:
        clarity = 0.0
    else:
        if clarity >= T2:
            clarity = 1.0
        else:
            clarity = (clarity - T1) / (T2 - T1)

    return clarity

def fix_pos(rect, height, width):
    x0 = rect[0]
    y0 = rect[1]
    x1 = rect[0] + rect[2]
    y1 = rect[1] + rect[3]
    x0 = min(max(x0, 0), width)
    y0 = min(max(y0, 0), height)
    x1 = min(max(x1, 0), width)
    y1 = min(max(y1, 0), height)
    fixed = [x0, y0, x1 - x0, y1 - y0]
    return fixed

def argmax(arr):
    size = len(arr.tolist())
    if size == 0:
        return 0
    arr_list = arr.reshape((1, -1)).tolist()
    flag = arr[0]
    max = 0
    for  i in range(size):
        if arr[i][0] > flag:
            flag = arr[i][0]
            max = i
    return max


def box_detect_postprocess(image, box_encodings, class_predictions):
    size = box_encodings.shape[1]
    result = []
    for i in range(size):
        label = argmax(class_predictions[0, i, 1:]) + 1
        score = class_predictions[0, i, int(label), 0]
        if score < 0.8:
            continue

        box_encoding = [0 for j in range(4)]
        box_encoding[0] = box_encodings[0,i,0,0]
        box_encoding[1] = box_encodings[0,i,0,1]
        box_encoding[2] = box_encodings[0,i,0,2]
        box_encoding[3] = box_encodings[0,i,0,3]

        anchor = [0 for j in range(4)]
        anchor[0] = fas_anchors[4 * i + 0]
        anchor[1] = fas_anchors[4 * i + 1]
        anchor[2] = fas_anchors[4 * i + 2]
        anchor[3] = fas_anchors[4 * i + 3]
        region = decode_box(anchor, box_encoding)

        x = region[0]
        y = region[1]
        width = region[2] - region[0]
        height = region[3] - region[1]

        box = {}
        box['label'] = int(label)
        box['score'] = score
        box['pos'] = []
        height = image.shape[0]
        width = image.shape[1]
        box['pos'].append(int(x * width))
        box['pos'].append(int(y * height))
        box['pos'].append(int(width * width))
        box['pos'].append(int(height * height))
        box['pos'] = fix_pos(box['pos'], height, width)
        result.append(box)
    return result

