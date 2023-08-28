import cv2


def increase_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


if __name__ == "__main__":
    import sys
    from os import path

    img = cv2.imread(sys.argv[1])
    basename = path.join(*sys.argv[1].split('.')[:-1])
    print(f'{basename}_brightened.png')
    cv2.imwrite(f'{basename}_brightened.png',
                increase_brightness(img, int(sys.argv[2])))
