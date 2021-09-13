# -*- coding: utf-8 -*-
import math
import copy

def prior_box_forward(box_dimension, img_height, img_width, tips):
    mean = []
    steps = [32, 64, 128]
    box_n = box_dimension.shape[0]
    prior_box_min_size = []
    prior_box_min_size.append([32, 64, 128])
    prior_box_min_size.append([256, ])
    prior_box_min_size.append([512, ])
    for k in range(box_n):
        f = box_dimension[k]
        min_sizes = prior_box_min_size[k]
        for i in range(int(f[0])):
            for j in range(int(f[1])):
                for min_size in min_sizes:
                    s_kx = float(min_size) / img_width
                    s_ky = float(min_size) / img_height
                    if min_size == 32:
                        fi = i
                        fj = j
                        dense_cx = [fj + 0.0, fj + 0.25, fj + 0.5, fj + 0.75]
                        dense_cy = [fi + 0.0, fi + 0.25, fi + 0.5, fi + 0.75]
                        dense_cx = [x * steps[k] / img_width for x in dense_cx]
                        dense_cy = [y * steps[k] / img_height for y in dense_cy]
                        for y in range(len(dense_cy)):
                            for x in range(len(dense_cx)):
                                cy = dense_cy[y]
                                cx = dense_cx[x]
                                mean.append((cx, cy, s_kx, s_ky))
                    elif min_size == 64:
                        fi = i
                        fj = j
                        dense_cx = [fj + 0.0, fj + 0.5]
                        dense_cy = [fi + 0.0, fi + 0.5]
                        dense_cx = [x * steps[k] / img_width for x in dense_cx]
                        dense_cy = [y * steps[k] / img_height for y in dense_cy]
                        for y in range(len(dense_cy)):
                            for x in range(len(dense_cx)):                        
                                cy = dense_cy[y]
                                cx = dense_cx[x]
                                mean.append((cx, cy, s_kx, s_ky))
                    else:
                        fi = i
                        fj = j
                        cx = (fj + 0.5) * steps[k] / img_width
                        cy = (fi + 0.5) * steps[k] / img_height
                        mean.append((cx, cy, s_kx, s_ky))

    return mean

def decode_single(location, priors, variance, score):
    box_data = [0 for i in range(5)]
    box_data[0] = priors[0] + location[0] * variance[0] * priors[2]
    box_data[1] = priors[1] + location[1] * variance[0] * priors[3]
    box_data[2] = priors[2] * math.exp(location[2] * variance[1])
    box_data[3] = priors[3] * math.exp(location[3] * variance[1])
    box_data[0] -= box_data[2] / 2
    box_data[1] -= box_data[3] / 2
    box_data[2] += box_data[0]
    box_data[3] += box_data[1]
    box_data[4] = score
    return box_data


def sort_box(boxes):
    sorted_boxes = []
    while(len(boxes)):
        max_prob = 0.0
        max_index = 0
        for i in range(len(boxes)):
            if boxes[i][4] > max_prob:
                max_index = i
                max_prob = boxes[i][4]
        sorted_boxes.append(copy.deepcopy(boxes[max_index]))
        del boxes[max_index]
    return sorted_boxes


def decode(location, priors, variance, confidence, threshold, top_k):
    boxes = []
    N = location.shape[1]
    assert N == len(priors)
    variance_data = variance
    cast_top_k = top_k
    for i in range(N):
        location_data = location[0][i]
        priors_data = priors[i]
        score = confidence[i][1]
        if score < threshold:
            continue
        box = decode_single(location_data, priors_data, variance_data, score)
        boxes.append(box)
    return sort_box(boxes)


def IoU(w1, w2):
    xOverlap = max(0, min(w1[2] - 1, w2[2] - 1) - max(w1[0], w2[0]) + 1)
    yOverlap = max(0, min(w1[3] - 1, w2[3] - 1) - max(w1[1], w2[1]) + 1)
    intersection = xOverlap * yOverlap
    w1_width = w1[2] - w1[0]
    w1_height = w1[3] - w1[1]
    w2_width = w2[2] - w2[0]
    w2_height = w2[3] - w2[1]
    unio = w1_width * w1_height + w2_width * w2_height - intersection
    return float(intersection) / unio

def nms_sorted(winList, threshold):
    if len(winList) == 0:
        return winList
    flag = [False for i in range(len(winList))]
    for i in range(len(winList)):
        if flag[i]:
            continue
        j = i + 1
        while j < len(winList):
            if IoU(winList[i], winList[j]) > threshold:
                flag[j] = True
            j += 1
    ret = []
    for i in range(len(winList)):
        if flag[i] == False:
            ret.append(winList[i])
    return ret