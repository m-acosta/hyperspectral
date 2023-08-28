import logging
from collections import namedtuple
from multiprocessing import Process, Queue
from sys import stdout

import numpy as np

import cv2
from hyperspectral_data_extractor import extract_wavelengths, load_image
from imageio import imsave

logging.basicConfig(stream=stdout, level=logging.DEBUG)


def pad_image(image, padding):
    r, c, ch = image.shape
    padded_image = np.zeros((r + 2 * padding, c, ch), dtype=np.float32)

    padded_image[padding:r + padding, :, :] = image[:, :, :]

    return padded_image


def load_and_extract(filename, wavelengths):
    return extract_wavelengths(load_image(filename), wavelengths)


def rotate(datacube, angle):
    r, c, _ = datacube.shape
    M = cv2.getRotationMatrix2D((c // 2, r // 2), angle, 1)
    return cv2.warpAffine(datacube, M, (c, r))


def find_crop(image):
    ret, thresh = cv2.threshold(image, 0, 255, 0)
    dilation = cv2.dilate(thresh, np.ones((5, 5)), iterations=1)
    contours, _ = cv2.findContours(
        dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bbox = namedtuple('bbox', ['x', 'y', 'w', 'h'])
    return bbox(*cv2.boundingRect(contours[-1]))


def crop(image, bbox):
    middle = image.shape[1] // 2
    return image[:, middle - bbox.w // 2:middle + bbox.w // 2, :]


def create_rgb(image):
    r, c, ch = image.shape

    rgb = np.zeros((r, c, 3), dtype=np.uint8)

    rgb[:, :, 0] = np.multiply(np.divide(image[:, :, 91], np.max(
        image[:, :, 91])), 255).astype(np.uint8)
    rgb[:, :, 1] = np.multiply(np.divide(image[:, :, 46], np.max(
        image[:, :, 46])), 255).astype(np.uint8)
    rgb[:, :, 2] = np.multiply(np.divide(image[:, :, 36], np.max(
        image[:, :, 36])), 255).astype(np.uint8)

    return rgb


def normalize(image):
    r, c, ch = image.shape

    normal = np.zeros((r, c, ch), dtype=np.uint8)

    for idx in range(ch):
        normal[:, :, idx] = np.multiply(np.divide(image[:, :, idx], np.max(
            image[:, :, idx])), 255).astype(np.uint8)

    return normal


def save_rgb(process):
    basename, rgb = process.get()

    logging.info(f'saving: {basename}.png')
    imsave(f'{basename}.png', rgb)

    process.close()


def save_raster(basename, raster):
    logging.info(f'saving: {basename}.npy')
    np.save(f'{basename}.npy', raster)


if __name__ == "__main__":
    import os
    from os import path
    import argparse
    import json

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('input', type=str,
                            help='directory with reflectance header files\
                                and datacubes')
        parser.add_argument('-o', '--output_dir', type=str, default=None,
                            help='directory of where rotated raster\
                                will be saved. Default: combined folder on\
                                    same level as input directory')
        parser.add_argument('-t', '--test', action='store_true', default=False,
                            help='perform a dry-run; prints which files will\
                                be read and written without actual modifications\
                                to the filesystem')
        parser.add_argument('-a', '--angle', type=float, default=130.8,
                            help='angle to rotate image in couter-clockwise\
                                direction. Default: 130.8')
        parser.add_argument('-p', '--padding', type=int, default=150,
                            help='amount of padding to add to image if not\
                                rotating by multiples of 90. Default: 150')
        parser.add_argument('-rgb', '--RGB_only', action='store_true', default=False,
                            help='create RGB image only no raster.')
        parser.add_argument('-v', '--verbose', action='store_true', default=False,
                            help='print out read and saved files.')
        return parser.parse_args()

    args = parse_args()

    in_dir = path.abspath(args.input)
    if not args.output_dir:
        out_dir = path.join('/', *in_dir.split('/')[:-1], 'rotated')
    else:
        out_dir = args.output_dir
    if args.test or args.verbose:
        logging.info(f'input folder: {in_dir}')
        logging.info(f'save folder: {out_dir}')

    files = [f for f in os.listdir(in_dir) if '.hdr' in f]
    if len(files) < 1:
        raise AssertionError(
            f'No header files found in {in_dir}')
    try:
        files = sorted(files, key=lambda x: int(x.split('.')[0]))
    except ValueError:
        files = sorted(files, key=lambda x: int(x.split('_')[1]))

    if not path.exists(out_dir):
        if args.test:
            logging.info(f'Folder {out_dir} would be created')
        else:
            os.makedirs(out_dir, mode=0o755, exist_ok=True)
            if args.verbose:
                logging.info(f'Folder {out_dir} created')

    with open('chosen_wavelengths.json') as fp:
        data = json.load(fp)
        wavelengths = [int(k) for k in data.keys()]

    best_bbox = None
    rgbs = []

    logging.info('determining largest bounding box')
    for fn in files:
        filename = path.join(in_dir, fn)
        if args.verbose:
            logging.info(f'reading: {filename}')

        img = load_and_extract(filename, [670, 541, 480])
        if args.angle % 90:
            img = pad_image(img, args.padding)
        rotated = rotate(img, args.angle)
        rgb = normalize(rotated)
        bbox = find_crop(rgb[:, :, 2])

        if best_bbox is None or bbox.w > best_bbox.w:
            best_bbox = bbox

        rgbs.append(rgb)

    logging.info('cropping and saving rgbs')
    if not args.test:
        q = Queue()

        for fn, rgb in zip(files, rgbs):
            basename = fn.split('.')[0]
            q.put((f'{out_dir}/{basename}', crop(rgb, best_bbox)))
            process = Process(target=save_rgb, args=(q,))
            process.start()

        q.close()
        q = None
        process = None
        rgbs = None

    if not args.RGB_only:
        logging.info('cropping and saving raster')
        for fn in files:
            filename = path.join(in_dir, fn)
            if args.verbose:
                logging.info(f'reading: {filename}')

            img = load_and_extract(filename, wavelengths)
            if args.angle % 90:
                img = pad_image(img, args.padding)
            raster = rotate(img, args.angle)
            img = None

            if not args.test:
                basename = fn.split('.')[0]
                save_raster(f'{out_dir}/{basename}', crop(raster, best_bbox))
                raster = None
