import numpy as np

from snippets import get_wavelength_index, load_wavelenth_file
from spectral import envi

load_wavelenth_file('wavelengths.json')


def load_image(filename):
    """Opens datacube into ndarray

    Args:
        filename(str): name of header file
    """

    try:
        datacube = envi.open(f'{filename}', filename.split('.')[0])
    except TypeError:
        print('TypeError: not ENVI header and binary')
        exit()
    except IOError:
        print('Cannot binary file')
        exit()
    except ValueError:
        print(f'Cannot open file: {filename}')
        exit()

    return datacube


def extract_wavelengths(datacube, wavelengths):
    ch = len(wavelengths)
    r, c, _ = datacube.shape
    raster = np.zeros((r, c, ch), dtype=np.float32)
    for idx, w in enumerate(wavelengths):
        raster[:, :, idx] = datacube[:, :,
                                     get_wavelength_index(w)].reshape((r, c))

    return raster


if __name__ == "__main__":
    import sys
    load_wavelenth_file('wavelengths.json')
    data = load_image(sys.argv[1])
    raster = extract_wavelengths(data, [670, 541, 480])
    raster *= 128
    d = extract_wavelengths(data, [679, 800, 900, 970])

    from imageio import imwrite
    imwrite('cropped.png', raster[1330:1415, 0:200, :])
    print(
        f'NDVI: {(np.sum(d[625:660, 120:170, 1]) - np.sum(d[625:660, 120:170, 0])) / (np.sum(d[625:660, 120:170, 1]) + np.sum(d[625:660, 120:170, 0]))}')
    print(
        f'WBI: {np.sum(d[625:660, 120:170, 2]) / np.sum(d[625:660, 120:170, 3])}')
