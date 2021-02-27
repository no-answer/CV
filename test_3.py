import cv2
import numpy


def histogram_equalization(image):
    if len(image.shape) == 2:                   # 单通道灰度图
        h, w = image.shape
        value_array = numpy.zeros(255)
        for i in range(h):
            for j in range(w):
                value_array[image[i, j]] += 1

        ratio_now = 0
        for i in range(255):
            value_array[i] /= h * w
            ratio_now += value_array[i]
            value_array[i] = round(255 * ratio_now)

        new_image = numpy.zeros((h, w))
        for i in range(h):
            for j in range(w):
                new_image[i, j] = value_array[image[i, j]]
        new_image = cv2.convertScaleAbs(new_image)
        return new_image

    h, w, d = image.shape
    value_arrays = numpy.zeros((image.shape[2], 256))
    for i in range(h):
        for j in range(w):
            for k in range(d):
                value_arrays[k, image[i, j, k]] += 1

    ratio_now = numpy.zeros(d)
    for i in range(d):
        for j in range(255):
            value_arrays[i, j] /= h * w
            ratio_now[i] += value_arrays[i, j]
            value_arrays[i][j] = round(255 * ratio_now[i])

    new_image = numpy.zeros((h, w, d))
    for i in range(h):
        for j in range(w):
            for k in range(d):
                new_image[i, j, k] = value_arrays[k, image[i, j, k]]
    cv2.normalize(new_image, new_image, 0, 255, cv2.NORM_MINMAX)
    new_image = cv2.convertScaleAbs(new_image)
    return new_image


def function_3():
    image_path = "image/1_1.png"
    image_source = cv2.imread(image_path)
    cv2.imshow("image source", image_source)
    histogram_equalization_image = histogram_equalization(image_source)
    cv2.imshow("histogram equalization image", histogram_equalization_image)

    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度图方式读取
    cv2.imshow("image gray", image_gray)
    histogram_equalization_image_gray = histogram_equalization(image_gray)
    cv2.imshow("histogram equalization image gray", histogram_equalization_image_gray)

    cv2.waitKey(0)