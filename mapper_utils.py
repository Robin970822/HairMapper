import bz2
import cv2
import copy
import numpy as np
from PIL import Image


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


def convert_cv2pil(input_image):
    input_image = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    return input_image


def convert_pil2cv(input_image):
    input_image = cv2.cvtColor(np.asarray(input_image), cv2.COLOR_RGB2BGR)
    return input_image


def warpaffine_contours(contours, affine_mat):
    new_contours = []
    for contour in contours:
        new_contour = copy.deepcopy(contour)
        for k in range(len(contour)):
            new_contour[k][0][0] = affine_mat[0][0] * contour[k][0][0] + \
                                   affine_mat[0][1] * contour[k][0][1] + \
                                   affine_mat[0][2]
            new_contour[k][0][1] = affine_mat[1][0] * contour[k][0][0] + \
                                   affine_mat[1][1] * contour[k][0][1] + \
                                   affine_mat[1][2]
        new_contours.append(new_contour)
    return new_contours
