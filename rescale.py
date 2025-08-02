import argparse
import os
from time import time
import json
import cv2

from skimage.transform import rescale
import numpy as np

from utils import (
        load_image, save_image, read_string, write_string,
        load_tsv, save_tsv)


def get_image_filename(prefix):
    file_exists = False
    for suffix in ['.jpg', '.png', '.tiff']:
        filename = prefix + suffix
        if os.path.exists(filename):
            file_exists = True
            break
    if not file_exists:
        raise FileNotFoundError('Image not found')
    return filename


def rescale_image(img, scale):
    if img.ndim == 2:
        scale = [scale, scale]
    elif img.ndim == 3:
        scale = [scale, scale, 1]
    else:
        raise ValueError('Unrecognized image ndim')
    img = rescale(img, scale, preserve_range=True)
    return img


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--image', action='store_true')
    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--locs', action='store_true')
    parser.add_argument('--radius', action='store_true')
    args = parser.parse_args()
    return args


def main():

    args = get_args()

    # # visium data
    # raw_img = cv2.imread(f'{args.prefix}image.tif')
    # hires_img = load_image(f'{args.prefix}spatial/tissue_hires_image.png')
    # with open(f'{args.prefix}spatial/scalefactors_json.json', 'r') as file:
    #     j = json.load(file)
    # tissue_hires_scalef = j["tissue_hires_scalef"]
    # pixel_size_raw = (raw_img.shape[0]/hires_img.shape[0])*tissue_hires_scalef
    # pixel_size = 0.5
    # scale = round(pixel_size_raw / pixel_size, 1)

    # her2st data
    raw_img = load_image('data/her2st/images/HE/H3.jpg')
    scale = 2.0

    if args.image:
        img = raw_img
        img = img.astype(np.float32)
        print(f'Rescaling image (scale: {scale:.2f})...')
        t0 = time()
        img = rescale_image(img, scale)
        print(int(time() - t0), 'sec')
        img = img.astype(np.uint8)
        save_image(img, args.prefix+'image-scaled.jpg')

    if args.mask:
        mask = load_image(args.prefix+'mask-raw.png')
        mask = mask > 0
        if mask.ndim == 3:
            mask = mask.any(2)
        print(f'Rescaling mask (scale: {scale:.3f})...')
        t0 = time()
        mask = rescale_image(mask.astype(np.float32), scale)
        print(int(time() - t0))
        mask = mask > 0.5
        save_image(mask, args.prefix+'mask-scaled.png')

    if args.locs:
        locs = load_tsv(args.prefix+'locs-raw.tsv')
        locs = locs * scale
        locs = locs.round().astype(int)
        save_tsv(locs, args.prefix+'locs.tsv')

    if args.radius:
        radius = float(read_string(args.prefix+'radius-raw.txt'))
        radius = radius * scale
        radius = np.round(radius).astype(int)
        write_string(radius, args.prefix+'radius.txt')


if __name__ == '__main__':
    main()
