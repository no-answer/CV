import cv2
import math
import numpy
import matplotlib.pyplot as plt


def fft_shift(image):
    h, w = image.shape
    new_image = numpy.zeros_like(image, dtype=numpy.complex)
    h1 = h // 2
    w1 = w // 2
    for i in range(h1):
        for j in range(w1):
            new_image[h1 - i - 1, w1 - j - 1] = image[h - i - 1, w - j - 1]
            new_image[h1 - i - 1, w1 + j] = image[h - i - 1, j]
            new_image[h1 + i, w1 - j - 1] = image[i, w - j - 1]
            new_image[h1 + i, w1 + j] = image[i, j]
    return new_image


def low_pass_filter(image, d0):
    h, w = image.shape
    new_image = numpy.zeros_like(image, dtype=numpy.complex)
    for i in range(h):
        for j in range(w):
            d = numpy.sqrt((i - h / 2) ** 2 + (j - w / 2) ** 2)
            if d < d0:
                new_image[i][j] = image[i][j]
            else:
                new_image[i][j] = numpy.complex(0)
    return new_image


def high_pass_filter(image, d0):
    h, w = image.shape
    new_image = numpy.zeros_like(image, dtype=numpy.complex)
    for i in range(h):
        for j in range(w):
            d = numpy.sqrt((i - h / 2) ** 2 + (j - w / 2) ** 2)
            if d < d0:
                new_image[i][j] = numpy.complex(0)
            else:
                new_image[i][j] = image[i][j]
    return new_image


def butterworth_low_pass_filter(image, d0, n):
    h, w = image.shape
    new_image = numpy.zeros_like(image, dtype=numpy.complex)
    for i in range(h):
        for j in range(w):
            d = numpy.sqrt((i - h / 2) ** 2 + (j - w / 2) ** 2)
            bt = 1 / (1 + (d / d0) ** (2 * n))
            new_image[i][j] = image[i][j] * bt
    return new_image


def butterworth_high_pass_filter(image, d0, n):
    h, w = image.shape
    new_image = numpy.zeros_like(image, dtype=numpy.complex)
    for i in range(h):
        for j in range(w):
            d = numpy.sqrt((i - h / 2) ** 2 + (j - w / 2) ** 2)
            bt = 1 / (1 + (d / d0) ** (2 * n))
            new_image[i][j] = image[i][j] * (1 - bt)
    return new_image


def function_5():
    image_path = "image/1_1.png"
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度图方式读取
    # cv2.imshow("image_gray", image_gray)

    fft = numpy.fft.fft2(image_gray)
    plt.subplot(3, 3, 1)
    plt.title('image_gray')
    plt.imshow(image_gray, cmap='gray')

    plt.subplot(3, 3, 2)
    plt.title('fft')
    plt.imshow(numpy.log(numpy.abs(fft)), cmap='gray')

    plt.subplot(3, 3, 3)
    plt.title('fft_shift')
    plt.imshow(numpy.log(numpy.abs(fft_shift(fft))), cmap='gray')

    plt.subplot(3, 3, 4)
    plt.title('low_pass_filter')
    temp = numpy.real(numpy.fft.ifft2(numpy.fft.ifftshift(low_pass_filter(numpy.fft.fftshift(fft), 10))))
    plt.imshow(temp, cmap='gray')

    plt.subplot(3, 3, 5)
    plt.title('butterworth_low_pass_filter')
    temp = numpy.real(numpy.fft.ifft2(numpy.fft.ifftshift(butterworth_low_pass_filter(numpy.fft.fftshift(fft), 10, 1))))
    plt.imshow(temp, cmap='gray')

    plt.subplot(3, 3, 7)
    plt.title('high_pass_filter')
    temp = numpy.real(numpy.fft.ifft2(numpy.fft.ifftshift(high_pass_filter(numpy.fft.fftshift(fft), 10))))
    plt.imshow(temp, cmap='gray')

    plt.subplot(3, 3, 8)
    plt.title('butterworth_high_pass_filter')
    temp = numpy.real(numpy.fft.ifft2(numpy.fft.ifftshift(butterworth_high_pass_filter(numpy.fft.fftshift(fft), 10, 1))))
    plt.imshow(temp, cmap='gray')

    plt.show()
    cv2.waitKey(0)