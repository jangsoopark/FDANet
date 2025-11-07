from skimage import transform
from skimage import io

from PIL import Image

from utils import resize

import numpy as np

import argparse
import math
import glob
import sys
import os

parser = argparse.ArgumentParser(description='SYSU-CD Change Data Patch Extraction')
parser.add_argument('--dataset-root', type=str,
                    default='/workspace/dataset/SYSU-CD/',
                    help='Original dataset root')
parser.add_argument('--experiment-root', type=str,
                    default='/workspace/dataset/SYSU-CD-experiment/',
                    help='Experiment dataset root')
parser.add_argument('--patch-size', type=int, default=256, help='dataset patch size')
parser.add_argument('--mode', type=str, default='val', help='train/val/test')
args = parser.parse_args()


def extract_patch(image, patch_size, stride):
    h, w, *_ = image.shape
    ph = math.ceil(h / patch_size)
    pw = math.ceil(w / patch_size)
    ps = patch_size

    patches = []
    for i in range(0, ph):
        sh = i * ps
        eh = (i + 1) * ps
        for j in range(0, pw):
            sw = j * ps
            ew = (j + 1) * ps

            _ph, _pw, c = image[sh: eh, sw: ew, :].shape

            sh, eh = (h - ps, h) if _ph < ps else (sh, eh)
            sw, ew = (w - ps, w) if _pw < ps else (sw, ew)

            _patch = image[sh: eh, sw: ew, :]
            assert ((ps, ps) == _patch.shape[:-1])
            patches.append(_patch)

    return patches


def augmentation(x1_path, x2_path, y_path, patch_size, stride, rotates, flips, scales, dst_root):
    x1 = io.imread(x1_path)
    x2 = io.imread(x2_path)
    y = io.imread(y_path)
    y = np.expand_dims(y, axis=2)

    name = os.path.splitext(os.path.basename(y_path))[0]

    h, w, _ = y.shape
    idx = 0
    for r in rotates:
        for f in flips:
            for s in scales:
                _x1 = transform.rotate(x1, angle=r) if r else x1
                _x1 = np.flip(_x1, axis=0) if f else _x1
                _x1 = resize.imresize(_x1, s) if s < 1 else _x1

                _x2 = transform.rotate(x2, angle=r) if r else x2
                _x2 = np.flip(_x2, axis=0) if f else _x2
                _x2 = resize.imresize(_x2, s) if s < 1 else _x2

                _y = transform.rotate(y, angle=r) if r else y
                _y = np.flip(_y, axis=0) if f else _y
                _y = resize.imresize(_y, s) if s < 1 else _y

                # print(r, f, s, _x1.shape, _x2.shape, _y.shape, (h, w))
                _x1_patches = extract_patch(_x1, patch_size, stride)
                _x2_patches = extract_patch(_x2, patch_size, stride)
                _y_patches = extract_patch(_y, patch_size, stride)

                assert (len(_x1_patches) == len(_x2_patches) and len(_x2_patches) == len(_y_patches))

                for p_x1, p_x2, p_y in zip(_x1_patches, _x2_patches, _y_patches):
                    io.imsave(
                        os.path.join(dst_root, f'A/{name}-{idx:010d}.png'),
                        p_x1,
                        plugin='tifffile', check_contrast=False
                    )
                    io.imsave(
                        os.path.join(dst_root, f'B/{name}-{idx:010d}.png'),
                        p_x2,
                        plugin='tifffile', check_contrast=False
                    )
                    io.imsave(
                        os.path.join(dst_root, f'label/{name}-{idx:010d}.png'),
                        p_y,
                        plugin='tifffile', check_contrast=False
                    )
                    idx += 1


# noinspection PyTypeChecker
def main():

    assert(args.mode in ['train', 'val', 'test'])

    scales = [1.00, ] if args.mode == 'train' else [1.00, ] #  1, 0.75, 0.5
    flips = [0, ] if args.mode == 'train' else [0, ]
    rotates = [0, ] if args.mode == 'train' else [0, ] # 0, 90, 180, 270
    os.makedirs(args.experiment_root, exist_ok=True)

    _image_a_pattern = os.path.join(args.dataset_root, f'{args.mode}/A/*.png')
    _image_b_pattern = os.path.join(args.dataset_root, f'{args.mode}/B/*.png')
    _label_pattern = os.path.join(args.dataset_root, f'{args.mode}/label/*.png')

    _image_a_list = glob.glob(_image_a_pattern)
    _image_b_list = glob.glob(_image_b_pattern)
    _label_list = glob.glob(_label_pattern)

    _image_a_list = sorted(_image_a_list, key=os.path.basename)
    _image_b_list = sorted(_image_b_list, key=os.path.basename)
    _label_list = sorted(_label_list, key=os.path.basename)

    print(len(_image_a_list), len(_image_b_list), len(_label_list))

    assert (len(_image_a_list) == len(_image_b_list) and len(_image_b_list) == len(_label_list) and len(_label_list))

    dst_root = os.path.join(args.experiment_root, args.mode)
    os.makedirs(os.path.join(dst_root, 'A'), exist_ok=True)
    os.makedirs(os.path.join(dst_root, 'B'), exist_ok=True)
    os.makedirs(os.path.join(dst_root, 'label'), exist_ok=True)

    for x1_path, x2_path, y_path in zip(_image_a_list, _image_b_list, _label_list):
        augmentation(x1_path, x2_path, y_path, args.patch_size, args.patch_size, rotates, flips, scales, dst_root)


if __name__ == '__main__':
    sys.exit(main())
