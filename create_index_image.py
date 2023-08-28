from imageio import imsave

wavelengths = [445, 480, 529, 541, 550, 570,
               670, 679, 699, 704, 748, 800, 900, 970]

wavelengths_dict = {}
for i, v in enumerate(wavelengths):
    wavelengths_dict[v] = i

EPSILON = 1000000


def RedEdge(NIR5, RED3):
    # return np.multiply(np.divide(NIR5 / RED3, np.max(
    #     NIR5 / RED3)), 255).astype(np.uint8)
    return NIR5 / (RED3 + EPSILON)


def NDVI(NIR1, RED3):
    # return np.multiply(np.divide((NIR1 - RED3) / (NIR1 + RED3), np.max(
    #     (NIR1 - RED3) / (NIR1 + RED3))), 255).astype(np.uint8)
    return (NIR1 - RED3) / (NIR1 + RED3 + EPSILON)


def save(name, a):
    # if np.min(a) < 0:
    #     a += np.min(a)

    imsave(name, ((a / np.max(a)) * 255))


if __name__ == '__main__':
    import sys
    import numpy as np

    img = np.load(sys.argv[1])
    r, c, _ = img.shape

    re = np.zeros((r, c, 1))
    re[:, :, 0] = RedEdge(img[:, :, wavelengths_dict[704]],
                          img[:, :, wavelengths_dict[670]])
    save(f'{sys.argv[1]}_rededge.png', re)

    ndvi = np.zeros((r, c, 1))
    ndvi[:, :, 0] = NDVI(img[:, :, wavelengths_dict[800]],
                         img[:, :, wavelengths_dict[670]])
    save(f'{sys.argv[1]}_ndvi.png', ndvi)
