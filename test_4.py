import cv2
import numpy
import math


def gaussian_window_1(sigma):
    window_size = math.floor(math.floor(6 * sigma - 1) / 2) * 2 + 1
    half = math.floor(window_size / 2)
    sum = 0
    window = numpy.zeros((window_size, window_size))
    for i in range(window_size):
        for j in range(window_size):
            x, y = i - half, j - half
            window[i, j] = math.pow(math.e, -((x * x + y * y) / (2 * sigma * sigma))) / (2 * math.pi * sigma * sigma)
            sum += window[i, j]
    for i in range(window_size):
        for j in range(window_size):
            window[i, j] /= sum
    return window


def gaussian_filter_1(image, sigma):
    h, w, d = image.shape
    new_image = numpy.zeros((h, w, d))
    window_size = math.floor(math.floor(6 * sigma - 1) / 2) * 2 + 1
    half = math.floor(window_size / 2)
    window = gaussian_window(sigma)
    for i in range(h):
        for j in range(w):
            for k in range(d):
                new_image_ijk = 0
                for ii in range(i - half - 1, i + half):
                    for jj in range(j - half - 1, j + half):
                        if ii < 0 or jj < 0 or ii >= h or jj >= w:
                            new_image_ijk += 0
                        else:
                            new_image_ijk += (image[ii, jj, k] * window[ii - i + half, jj - j + half])
                new_image[i, j, k] = math.floor(new_image_ijk)
    cv2.normalize(new_image, new_image, 0, 255, cv2.NORM_MINMAX)
    new_image = cv2.convertScaleAbs(new_image)
    return new_image


def gaussian_window(sigma):
    window_size = math.floor(math.floor(6 * sigma - 1) / 2) * 2 + 1
    half = math.floor(window_size / 2)
    window = numpy.zeros(window_size)
    sum = 0
    for i in range(window_size):
        x = i - half
        window[i] = math.pow(math.e, -(x * x) / (2 * sigma * sigma)) / (math.sqrt(2 * math.pi) * sigma)
        sum += window[i]
    for i in range(window_size):
        window[i] /= sum
    return window


def gaussian_filter(image, sigma):
    h, w, d = image.shape
    new_image_1 = numpy.zeros((h, w, d))
    new_image_2 = new_image_1
    window_size = math.floor(math.floor(6 * sigma - 1) / 2) * 2 + 1
    half = math.floor(window_size / 2)
    window = gaussian_window(sigma)
    for i in range(h):
        for j in range(w):
            for k in range(d):
                new_image_ijk = 0
                for index in range(i - half - 1, i + half):
                    if index < 0 or index >= h:
                        new_image_ijk += 0
                    else:
                        new_image_ijk += image[index, j, k] * window[index - i + half]
                new_image_1[i, j, k] = new_image_ijk

    for i in range(h):
        for j in range(w):
            for k in range(d):
                new_image_ijk = 0
                for index in range(j - half - 1, j + half):
                    if index < 0 or index >= w:
                        new_image_ijk += 0
                    else:
                        new_image_ijk += new_image_1[i, index, k] * window[index - j + half]
                new_image_2[i, j, k] = math.floor(new_image_ijk)
    cv2.normalize(new_image_2, new_image_2, 0, 255, cv2.NORM_MINMAX)
    new_image = cv2.convertScaleAbs(new_image_2)
    return new_image


def median_filter(image, window_size):
    h, w, d = image.shape
    new_image = numpy.zeros((h, w, d))
    half = int((window_size - 1) / 2)
    for i in range(h):
        for j in range(w):
            for k in range(d):
                array = numpy.zeros(window_size * window_size)
                index = 0
                for ii in range(i - half - 1, i + half):
                    for jj in range(j - half - 1, j + half):
                        if ii < 0 or jj < 0 or ii >= h or jj >= w:
                            array[index] = 0
                            index += 1
                        else:
                            array[index] = image[ii, jj, k]
                            index += 1
                array = numpy.sort(array)
                new_image[i, j, k] = array[int((window_size * window_size - 1) / 2)]

    cv2.normalize(new_image, new_image, 0, 255, cv2.NORM_MINMAX)
    new_image = cv2.convertScaleAbs(new_image)
    return new_image


def mean_filter(image, window_size):
    h, w, d = image.shape
    image_sum = numpy.zeros((h, w, d))
    for i in range(h):
        for j in range(w):
            for k in range(d):
                if i == 0 or j == 0:
                    image_sum[i, j, k] = image[i, j, k]
                else:
                    image_sum[i, j, k] = (image_sum[i - 1, j, k] + image_sum[i, j - 1, k] -
                                          image_sum[i - 1, j - 1, k] + image[i, j, k])

    half = int((window_size - 1) / 2)
    new_image = numpy.zeros((h, w, d))

    for i in range(h):
        for j in range(w):
            for k in range(d):
                if i == 0 or j == 0:
                    new_image[i, j, k] = image_sum[i, j, k] / window_size / window_size
                else:
                    new_image[i, j, k] = (1 / window_size / window_size) * (
                                image_sum[i + half, j + half, k] + image_sum[i - half - 1, j - half - 1, k] -
                                image_sum[i + half, j - half - 1, k] - image_sum[i - half - 1, j + half, k])

    cv2.normalize(new_image, new_image, 0, 255, cv2.NORM_MINMAX)
    new_image = cv2.convertScaleAbs(new_image)
    return new_image


def function_4():
    image_path = "image/4_1.png"
    image_source = cv2.imread(image_path)
    # cv2.imshow("image source", image_source)

    # image_mean_filter = mean_filter(image_source, 3)
    # cv2.imshow("image mean filter", image_mean_filter)

    # image_median_filter = median_filter(image_source, 3)
    # cv2.imshow("image median filter", image_median_filter)

    image_gaussion_filter = gaussian_filter(image_source, 3)
    cv2.imshow("image gaussian filter", image_gaussion_filter)

    cv2.waitKey(0)