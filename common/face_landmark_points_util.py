# -*- coding: utf-8 -*-
import numpy as np

m_origin_patch = [15, 15]
m_origin = [112, 112]

class HypeShape:
    def __init__(self, shape):
        self.m_shape = shape
        self.m_weights = [0]*len(self.m_shape)
        size = len(self.m_shape)
        self.m_weights[size - 1] = self.m_shape[size - 1]
        for times in range(size - 1):
             self.m_weights[size - 1 - times - 1] =  self.m_weights[size - 1 - times] * self.m_shape[size - 1 - times - 1]

    def to_index(self, coordinate):
        if len(coordinate) == 0:
            return 0
        size = len(coordinate)
        weight_start = len(self.m_weights) - size + 1
        index = 0
        for times in range(size - 1):
            index += self.m_weights[weight_start + times] * coordinate[times]
        index += coordinate[size - 1]
        return index


def shape_index_process(feat_data, pos_data):
    feat_h = feat_data.shape[2]
    feat_w = feat_data.shape[3]

    landmarkx2 = pos_data.shape[1]
    x_patch_h = int( m_origin_patch[0] * feat_data.shape[2] / float( m_origin[0] ) + 0.5 )
    x_patch_w = int( m_origin_patch[1] * feat_data.shape[3] / float( m_origin[1] ) + 0.5 )

    feat_patch_h = x_patch_h
    feat_patch_w = x_patch_w

    num = feat_data.shape[0]
    channels = feat_data.shape[1]

    r_h = ( feat_patch_h - 1 ) / 2.0
    r_w = ( feat_patch_w - 1 ) / 2.0
    landmark_num = int(landmarkx2 * 0.5)

    pos_offset = HypeShape([pos_data.shape[0], pos_data.shape[1]])
    feat_offset = HypeShape([feat_data.shape[0], feat_data.shape[1], feat_data.shape[2], feat_data.shape[3]])
    nmarks = int( landmarkx2 * 0.5 )
    out_shape = [feat_data.shape[0], feat_data.shape[1], x_patch_h, nmarks, x_patch_w]
    out_offset = HypeShape([feat_data.shape[0], feat_data.shape[1], x_patch_h, nmarks, x_patch_w])
    buff = np.zeros(out_shape)
    zero = 0

    buff = buff.reshape((-1))
    pos_data = pos_data.reshape((-1))
    feat_data = feat_data.reshape((-1))

    for i in range(landmark_num):
        for n in range(num):
            # coordinate of the first patch pixel, scale to the feature map coordinate
            y = int( pos_data[pos_offset.to_index( [n, 2 * i + 1] )] * ( feat_h - 1 ) - r_h + 0.5 )
            x = int( pos_data[pos_offset.to_index( [n, 2 * i] )] * ( feat_w - 1 ) - r_w + 0.5 )

            for c in range(channels):
                for ph in range(feat_patch_h):
                    for pw in range(feat_patch_w):
                        y_p = y + ph
                        x_p = x + pw
                        # set zero if exceed the img bound
                        if y_p < 0 or y_p >= feat_h or x_p < 0 or x_p >= feat_w:
                            buff[out_offset.to_index( [n, c, ph, i, pw] )] = zero
                        else:
                            buff[out_offset.to_index( [n, c, ph, i, pw] )] = feat_data[feat_offset.to_index( [n, c, y_p, x_p] )]

    return buff.reshape((1,-1,1,1)).astype(np.float32)