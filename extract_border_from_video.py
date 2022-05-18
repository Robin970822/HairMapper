import os
import cv2
import sys
import time
import torch
import argparse

import numpy as np
from tqdm import tqdm
from argparse import Namespace
from tensorflow.keras.utils import get_file

sys.path.append("encoder4editing")
from encoder4editing.models.psp import pSp

from styleGAN2_ada_model.stylegan2_ada_generator import StyleGAN2adaGenerator
from mapper.networks.level_mapper import LevelMapper
from classifier.src.feature_extractor.hair_mask_extractor import get_parsingNet

from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector

from mapper_utils import unpack_bz2, convert_cv2pil, convert_pil2cv, warpaffine_contours
from extract_border_from_image import extract_head_border_from_image, ensure_dir, ensure_sub_dir

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='./stylegan-encoder/video/video_huangting_1.MP4',
                        help='Directory of test data.')
    parser.add_argument('--orientation', default='top')
    parser.add_argument("--store_image",
                        help="if set, store the edited images",
                        action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    video_path = args.video_path
    orientation = args.orientation
    store_image = args.store_image
    # landmarks_detector
    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))
    landmarks_detector = LandmarksDetector(landmarks_model_path)

    ## head border
    encode_model_path = "./ckpts/e4e_ffhq_encode.pt"
    ckpt = torch.load(encode_model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = encode_model_path
    # opts['device'] = 'cpu:0'
    opts = Namespace(**opts)
    encode_net = pSp(opts).eval().cuda()

    model_name = 'stylegan2_ada'
    gan_model = StyleGAN2adaGenerator(model_name, logger=None, truncation_psi=0.75)
    mapper = LevelMapper(input_dim=512).eval().cuda()
    ckpt = torch.load('./mapper/checkpoints/final/best_model.pt')
    alpha = float(ckpt['alpha']) * 1.2
    mapper.load_state_dict(ckpt['state_dict'], strict=True)
    parsingNet = get_parsingNet(save_pth='./ckpts/face_parsing.pth')

    if video_path is not None:
        print("get video in, save video out")
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            print("open video file fail")
            exit(-1)
        video_name = (video_path.split('/')[-1]).split('.')[0]
    else:
        print("open camera")
        capture = cv2.VideoCapture(0)
        video_name = 'camera'
        if not capture.isOpened():
            print("open video file fail")
            exit(-1)

    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    save_video_path = f'result/{video_name}_{now}.mp4'
    ensure_sub_dir(save_video_path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if orientation == 'right' or orientation == 'left':
        temp = width
        width = height
        height = temp
    writer = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc("D", "I", "V", "X"), fps, (width * 2, height))
    frames_num = int(capture.get(7))

    if store_image:
        face_img_dir = f'./result/raw_image_{video_name}'
        aligned_img_dir = f'./result/aligned_image_{video_name}'
        image_out_path = f'result/{video_name}_{now}/out'
        crop_out_path = f'result/{video_name}_{now}/crop'
        edit_out_path = f'result/{video_name}_{now}/edit'
        mask_out_path = f'result/{video_name}_{now}/mask'
        list(map(lambda x: ensure_dir(x),
                 [face_img_dir, aligned_img_dir, image_out_path, crop_out_path, edit_out_path, mask_out_path]))

    for idx in tqdm(range(frames_num)):
        ori_face_img_name = '{:04d}.png'.format(idx)
        if not capture.isOpened():
            break
        ret, img = capture.read()
        if orientation == 'right':
            img = cv2.transpose(img)
            img = cv2.flip(img, 0)
        if orientation == 'left':
            img = cv2.transpose(img)
            img = cv2.flip(img, 1)

        out_img = img.copy()
        contour_img = img.copy()
        for i, face_landmarks in enumerate(landmarks_detector.get_landmarks_img(img), start=1):
            for k in range(len(face_landmarks)):
                cv2.circle(out_img, (int(face_landmarks[k][0]), int(face_landmarks[k][1])), 2, (0, 0, 255), -1)
            face_img_name = '{:04d}_{:04d}.png'.format(idx, i)
            aligned_img_name = 'aligned_' + face_img_name
            aligned_face_path = os.path.join(aligned_img_dir, aligned_img_name)
            aligned_img, affine_mat = image_align(convert_cv2pil(img), aligned_face_path, face_landmarks,
                                                  store_image=store_image)
            try:
                contours, edited_img, face_mask, hair_mask = extract_head_border_from_image(aligned_img, encode_net,
                                                                                            gan_model, mapper, alpha,
                                                                                            parsingNet)
                inverse_affine_mat = cv2.invertAffineTransform(affine_mat)
                if store_image:
                    crop_img = convert_pil2cv(aligned_img)
                    crop_img = cv2.drawContours(crop_img, contours, -1, (0, 0, 255), 5)
                contours = warpaffine_contours(contours, inverse_affine_mat)
                contour_img = cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 5)

                if store_image:
                    list(map(lambda x, y, z: cv2.imwrite(os.path.join(x, y + face_img_name), z),
                             [crop_out_path, edit_out_path, mask_out_path, mask_out_path],
                             ['crop_', 'edit_', 'face_', 'hair_'],
                             [crop_img, edited_img, face_mask, hair_mask]))
            except:
                print('cannot extract contours')
                continue
        if store_image:
            face_img_path = os.path.join(face_img_dir, 'raw_' + ori_face_img_name)
            cv2.imwrite(face_img_path, img)
        img = np.concatenate([out_img, contour_img], 1)
        if store_image:
            out_path = os.path.join(image_out_path, 'out_' + ori_face_img_name)
            cv2.imwrite(out_path, img)
        writer.write(img)
    writer.release()
