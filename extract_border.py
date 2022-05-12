import cv2
import glob
import os

from tqdm import tqdm
from classifier.src.feature_extractor.hair_mask_extractor import get_app_mask, get_parsingNet


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_image(name, image_dir='./test_data/origin', prefix=''):
    f_path_png = os.path.join(image_dir, '{}{}.png'.format(prefix, name))
    f_path_jpg = os.path.join(image_dir, '{}{}.jpg'.format(prefix, name))
    if os.path.exists(f_path_png):
        return f_path_png
    elif os.path.exists(f_path_jpg):
        return f_path_jpg


def draw_contour_with_mask(img, mask, color=(0, 0, 255), line_width=5):
    contours, _ = cv2.findContours(mask[:, :, 0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c_img = img.copy()
    c_img = cv2.drawContours(c_img, contours, -1, color, line_width)
    return c_img


if __name__ == '__main__':

    parsingNet = get_parsingNet(save_pth='./ckpts/face_parsing.pth')

    data_root = './test_data'
    code_dir = os.path.join(data_root, 'code')
    edit_dir = os.path.join(data_root, 'mapper_edit')

    face_dir = os.path.join(data_root, 'parsing_face')
    face_mask_dir = os.path.join(data_root, 'parsing_face_mask')

    mkdir(face_dir)
    mkdir(face_mask_dir)

    code_list = glob.glob(os.path.join(code_dir, '*.npy'))
    total_num = len(code_list)

    for index in tqdm(list(range(total_num))):
        code_path = code_list[index]
        name = os.path.basename(code_path)[:-4]

        origin_img_path = get_image(name)
        edit_img_path = get_image(name, edit_dir, prefix='edit_')

        origin_img = cv2.imread(origin_img_path)
        edit_img = cv2.imread(edit_img_path)

        face_mask, _, hair_mask = get_app_mask(img_path=edit_img, net=parsingNet, include_hat=True, include_ear=True)
        mask = face_mask + hair_mask

        c_edit_img = draw_contour_with_mask(edit_img, mask)
        if not origin_img.shape == edit_img.shape:
            image_height, image_width = origin_img.shape[:2]
            mask = cv2.resize(mask, (image_width, image_height))
        c_origin_img = draw_contour_with_mask(origin_img, mask)

        cv2.imwrite(os.path.join(face_dir, 'c_{}.png'.format(name)), c_edit_img)
        cv2.imwrite(os.path.join(face_dir, 'o_{}.png'.format(name)), c_origin_img)
        cv2.imwrite(os.path.join(face_mask_dir, 'm_{}.png'.format(name)), mask)
