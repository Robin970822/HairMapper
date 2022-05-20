import sys
import cv2
import torch
import glob
import os
import argparse
import numpy as np
import torchvision.transforms as transforms

from tqdm import tqdm
from argparse import Namespace
from PIL import Image

sys.path.append("encoder4editing")
sys.path.append("")
from mapper_utils import convert_cv2pil
from encoder4editing.models.psp import pSp
from styleGAN2_ada_model.stylegan2_ada_generator import StyleGAN2adaGenerator
from mapper.networks.level_mapper import LevelMapper
from classifier.src.feature_extractor.hair_mask_extractor import get_hair_mask, get_app_mask, get_parsingNet


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./test_data',
                        help='Directory of test data.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate for optimization.')
    parser.add_argument('--num_iterations', type=int, default=100,
                        help='Number of optimization iterations.')

    parser.add_argument('--loss_weight_feat', type=float, default=3e-5,
                        help='The perceptual loss weight.')
    parser.add_argument('--loss_weight_id', type=float, default=1.0,
                        help='The facial identity loss weight')
    parser.add_argument("--remain_ear",
                        help="if set, remain ears in the original image",
                        action="store_true")
    parser.add_argument("--diffuse",
                        help="if set, perform an additional diffusion method",
                        action="store_true")
    parser.add_argument("--store_image",
                        help="if set, store the edited images",
                        action="store_true")

    parser.add_argument('--dilate_kernel_size', type=int, default=50,
                        help='dilate kernel size')

    parser.add_argument('--blur_kernel_size', type=int, default=30,
                        help='blur kernel size')

    parser.add_argument('--truncation_psi', type=float, default='0.75')
    return parser.parse_args()


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def ensure_sub_dir(path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_models():
    ## head border
    encode_model_path = "./HairMapper/ckpts/e4e_ffhq_encode.pt"
    ckpt = torch.load(encode_model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = encode_model_path
    opts = Namespace(**opts)
    encode_net = pSp(opts).eval().cuda()

    model_name = 'stylegan2_ada'
    gan_model = StyleGAN2adaGenerator(model_name, logger=None, truncation_psi=0.75)
    mapper = LevelMapper(input_dim=512).eval().cuda()
    ckpt = torch.load('./HairMapper/mapper/checkpoints/final/best_model.pt')
    alpha = float(ckpt['alpha']) * 1.2
    mapper.load_state_dict(ckpt['state_dict'], strict=True)
    parsingNet = get_parsingNet(save_pth='./HairMapper/ckpts/face_parsing.pth')
    return encode_net, gan_model, mapper, alpha, parsingNet


def run_on_batch(inputs, net):
    latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
    return latents


def encode_image(input_image, net):
    input_image = input_image.copy()
    img_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    transformed_image = img_transforms(input_image)
    with torch.no_grad():
        latents = run_on_batch(transformed_image.unsqueeze(0), net)
        latent = latents[0].cpu().numpy()
        latent = np.reshape(latent, (1, 18, 512))
        return latent


def edit_image(latent, model, mapper, alpha):
    # editing latent code
    mapper_input = latent.copy()
    mapper_input_tensor = torch.from_numpy(mapper_input).cuda().float()
    edited_latent_codes = latent
    edited_latent_codes[:, :8, :] += alpha * mapper(mapper_input_tensor).to('cpu').detach().numpy()
    # editing image
    outputs = model.easy_style_mixing(latent_codes=edited_latent_codes,
                                      style_range=range(7, 18),
                                      style_codes=latent,
                                      mix_ratio=0.8,
                                      latent_space_type='wp'
                                      )
    edited_img = outputs['image'][0][:, :, ::-1]
    return edited_img


def mix_images(origin_img, edited_img, parsingNet):
    # --remain_ear: preserve the ears in the original input image.
    hair_mask = get_hair_mask(img_path=origin_img, net=parsingNet, include_hat=True, include_ear=True)

    mask_dilate = cv2.dilate(hair_mask, kernel=np.ones((50, 50), np.uint8))
    mask_dilate_blur = cv2.blur(mask_dilate, ksize=(10, 10))
    mask_dilate_blur = (hair_mask + (255 - hair_mask) / 255 * mask_dilate_blur).astype(np.uint8)

    face_mask = 255 - mask_dilate_blur

    index = np.where(face_mask > 0)
    cy = (np.min(index[0]) + np.max(index[0])) // 2
    cx = (np.min(index[1]) + np.max(index[1])) // 2
    center = (cx, cy)

    image_height, image_width = origin_img.shape[:2]
    edited_img = cv2.resize(edited_img, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
    try:
        mixed_clone = cv2.seamlessClone(origin_img, edited_img, face_mask[:, :, 0], center, cv2.NORMAL_CLONE)
    except:
        mask = (hair_mask / 255).astype('uint8')
        mixed_clone = origin_img * (1 - mask) + edited_img * mask
    return mixed_clone, hair_mask, face_mask


def extract_head_border_from_image(origin_img, encode_net, model, mapper, alpha, segmentors, flip=False, mix=False):
    if isinstance(origin_img, np.ndarray):
        origin_img = convert_cv2pil(origin_img)
    if flip:
        origin_img = origin_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    # latent code
    latent = encode_image(origin_img, encode_net)
    # editing image
    edited_img = edit_image(latent, model, mapper, alpha)
    # extract head border
    if mix:
        edited_img = mix_images(origin_img, edited_img, segmentors)
    if isinstance(segmentors, dict):
        face_segmentor = segmentors['face_segmentor']
        hair_segmentor = segmentors['hair']
        face_mask = face_segmentor.segment(edited_img)
        hair_mask = hair_segmentor.segment(edited_img)
        mask = np.bitwise_or(face_mask, hair_mask).astype('uint8')
    else:
        face_mask, _, hair_mask = get_app_mask(img_path=edited_img, net=segmentors, include_hat=True, include_ear=True)
        mask = face_mask + hair_mask
        mask = mask[:, :, 0]
    if flip:
        [mask, edited_img, face_mask, hair_mask] = list(
            map(lambda x: cv2.flip(x, 1), [mask, edited_img, face_mask, hair_mask]))
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, edited_img, face_mask, hair_mask


def run():
    args = parse_args()
    encode_model_path = "ckpts/e4e_ffhq_encode.pt"
    ckpt = torch.load(encode_model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = encode_model_path
    opts = Namespace(**opts)
    encode_net = pSp(opts)
    encode_net.eval()
    encode_net.cuda()

    model_name = 'stylegan2_ada'
    gan_model = StyleGAN2adaGenerator(model_name, logger=None, truncation_psi=args.truncation_psi)
    mapper = LevelMapper(input_dim=512).eval().cuda()
    ckpt = torch.load('./mapper/checkpoints/final/best_model.pt')
    alpha = float(ckpt['alpha']) * 1.2
    mapper.load_state_dict(ckpt['state_dict'], strict=True)
    parsingNet = get_parsingNet(save_pth='./ckpts/face_parsing.pth')

    origin_img_dir = os.path.join(args.data_dir, 'origin')
    if args.store_image:
        border_dir = os.path.join(args.data_dir, 'head_border')
        ensure_dir(border_dir)

    for file_path in tqdm(glob.glob(os.path.join(origin_img_dir, '*.png')) + glob.glob(
            os.path.join(origin_img_dir, '*.jpg'))):
        name = os.path.basename(file_path)[:-4]

        origin_img = cv2.imread(file_path)
        origin_img = convert_cv2pil(origin_img)
        contours = extract_head_border_from_image(origin_img, encode_net, gan_model, mapper, alpha, parsingNet)
        if args.store_image:
            c_img = origin_img.copy()
            c_img = cv2.drawContours(c_img, contours, -1, (0, 0, 255), 5)
            cv2.imwrite(os.path.join(border_dir, 'b_{}.png'.format(name)), c_img)


if __name__ == '__main__':
    run()
