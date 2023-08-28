import json
import logging
import os
from multiprocessing import Process, Queue
from os import path
from sys import stdout

import imageio
import numpy as np
import pandas as pd

logging.basicConfig(stream=stdout, level=logging.INFO)


def create_rgb(image):
    r, c, _ = image.shape

    rgb = np.zeros((r, c, 3), dtype=np.uint8)

    rgb[:, :, 0] = np.multiply(np.divide(image[:, :, 91], np.max(
        image[:, :, 91])), 255).astype(np.uint8)
    rgb[:, :, 1] = np.multiply(np.divide(image[:, :, 46], np.max(
        image[:, :, 46])), 255).astype(np.uint8)
    rgb[:, :, 2] = np.multiply(np.divide(image[:, :, 36], np.max(
        image[:, :, 36])), 255).astype(np.uint8)

    return rgb


def save_image(process, test, color):
    save_name, cropped = process.get()

    if test:
        logging.info('{:>100}.npy height: {:>3} width: {:>3} channels: {:>4}'.format(
            save_name, *cropped.shape))
    else:
        np.save(save_name, cropped)
        logging.info(f'saved (raster): {save_name}.npy')

    if color:
        rgb = create_rgb(cropped)
        cropped = None
        if test:
            logging.info('{:>100}.png height: {:>3} width: {:>3} channels: {:>4}'.format(
                save_name, *rgb.shape))
        else:
            imageio.imsave(
                f'{save_name}.png', rgb)
            logging.info(f'saved (rgb): {save_name}.png')
    else:
        cropped = None

    rgb = None

    process.close()


def crop(file_path, row_order_path, *args, **kwargs):

    # Open json file, which returns a list of annotation dictionaries
    with open(file_path, 'r') as f:
        annotations_list = json.load(f)
        row_order = pd.read_csv(row_order_path, header=None)

    output_path = path.join(
        '/', *path.abspath(file_path).split('/')[:-1], 'cropped')
    logging.info(f'output path: {output_path}')
    os.makedirs(output_path, mode=0o755, exist_ok=True)

    # make sure both row order and number annotations are the same before
    # cropping into single images
    num_annotations = 0
    for d in annotations_list:
        num_annotations += len(d['annotations'])
    logging.debug(
        f'annotations: {num_annotations}\t row_order: {len(row_order)}')
    assert num_annotations == len(row_order)

    row_order_iterator = row_order[0].iteritems()
    # keys: annotations, class, filename
    for d in annotations_list:
        annotations = d['annotations']
        if not len(annotations):
            continue

        images_path = path.join(
            '/', *path.abspath(file_path).split('/')[:-1], *d['filename'].split('/')[:-1])
        raster_name = f'{d["filename"].split(".")[0].split("/")[-1]}.npy'

        logging.info(f'reading: {path.join(images_path, raster_name)}')
        raster = np.load(path.join(images_path, raster_name))

        q = Queue()
        for anno, save_name in zip(annotations, row_order_iterator):
            x, y, w, h = (anno['x'], anno['y'], anno['width'], anno['height'])
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            cropped = raster[y: y + h, x: x + w, :]

            q.put((path.join(output_path, save_name[1]), cropped))

            process = Process(target=save_image, args=(
                q, kwargs.get('test', False), kwargs.get('color', False)))

            process.start()
        q.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('annotations', type=str)
    parser.add_argument('row_order', type=str)
    parser.add_argument('-T', default=False, action='store_true')
    parser.add_argument('-rgb', default=False, action='store_true')
    args = parser.parse_args()

    crop(args.annotations, args.row_order, test=args.T, color=args.rgb)
