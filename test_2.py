import cv2
import numpy
import math


def resize(image, scale_x, scale_y):
    h, w, d = image.shape
    re_h = math.ceil(h * scale_y)
    re_w = math.ceil(w * scale_x)

    new_image = numpy.zeros((re_h, re_w, d))
    for i in range(re_h):
        for j in range(re_w):
            for k in range(d):
                x = j / scale_x     # 求对应的原始坐标
                y = i / scale_y

                mark1 = (0 if scale_x <= 1 else 2)
                mark2 = (0 if scale_y <= 1 else 2)
                p1 = (min(math.floor(x), w-mark1), min(math.floor(y), h-mark2))
                p2 = (p1[0], p1[1] + 1)
                p3 = (p1[0] + 1, p1[1])
                p4 = (p1[0] + 1, p1[1] + 1)
                p13 = (x - p1[0]) * image[p1[1], p1[0], k] + (p3[0] - x) * image[p3[1], p3[0], k]
                p24 = (x - p2[0]) * image[p2[1], p2[0], k] + (p4[0] - x) * image[p4[1], p4[0], k]
                new_image[i, j, k] = (y - p1[1]) * p13 + (p2[1] - y) * p24
                # print(new_image[i, j, k])

    cv2.normalize(new_image, new_image, 0, 255, cv2.NORM_MINMAX)
    new_image = cv2.convertScaleAbs(new_image)
    return new_image


def reshape(image):
    h, w, d = image.shape
    new_image = numpy.zeros((h, w, d))
    
    for i in range(h):
        for j in range(w):
            x = (j - 0.5 * w) / (0.5 * w)
            y = (i - 0.5 * h) / (0.5 * h)
            r = math.sqrt(x * x + y * y)
            new_x, new_y = x, y
            if r < 1:
                theta = (1 - r) * (1 - r)
                new_x = math.cos(theta) * x - math.sin(theta) * y
                new_y = math.sin(theta) * x + math.cos(theta) * y

            new_x = round(new_x * 0.5 * w + 0.5 * w)
            new_y = round(new_y * 0.5 * h + 0.5 * h)
            new_image[i, j] = image[new_y, new_x]
    cv2.normalize(new_image, new_image, 0, 255, cv2.NORM_MINMAX)
    new_image = cv2.convertScaleAbs(new_image)
    return new_image


def function_2():
    image_path = "image/1_1.png"
    image_source = cv2.imread(image_path)
    cv2.imshow("image source", image_source)

    image_resize = resize(image_source, 0.3, 0.5)
    cv2.imshow("image resize 0.3 0.5", image_resize)

    image_reshape = reshape(image_source)
    cv2.imshow("image reshape", image_reshape)

    cv2.waitKey(0)