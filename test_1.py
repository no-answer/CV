import cv2
import math
import numpy
import pyglet


def log_trans(c, image):
    h, w, d = image.shape
    new_image = numpy.zeros(image.shape, dtype=numpy.uint8)
    for i in range(h):
        for j in range(w):
            for k in range(d):
                new_image[i, j, k] = math.floor(min(1, c * (math.log(1.0 + image[i, j, k]/255.0, 2)))*255)
    return new_image


def gamma_trans(c, gamma, image):
    h, w, d = image.shape
    new_image = numpy.zeros((h, w, d), dtype=numpy.float32)
    for i in range(h):
        for j in range(w):
            new_image[i, j, 0] = c * math.pow(image[i, j, 0], gamma)
            new_image[i, j, 1] = c * math.pow(image[i, j, 1], gamma)
            new_image[i, j, 2] = c * math.pow(image[i, j, 2], gamma)
    cv2.normalize(new_image, new_image, 0, 255, cv2.NORM_MINMAX)
    new_image = cv2.convertScaleAbs(new_image)
    return new_image


def log_trans_gray(c, image):
    h, w = image.shape
    new_image = numpy.zeros((h, w), dtype=numpy.uint8)
    for i in range(h):
        for j in range(w):
            new_image[i, j] = math.floor(min(1, c * (math.log(1.0 + image[i, j]/255.0, 2)))*255)
    return new_image


def gamma_trans_gray(c, gamma, image):
    h, w = image.shape
    new_image = numpy.zeros((h, w))
    for i in range(h):
        for j in range(w):
            new_image[i, j] = c * math.pow(image[i, j], gamma)
    cv2.normalize(new_image, new_image, 0, 255, cv2.NORM_MINMAX)
    new_image = cv2.convertScaleAbs(new_image)
    return new_image


def contrast_trans(c, image):
    h, w, d = image.shape
    new_image = numpy.zeros((h, w, d), dtype = numpy.float32)
    average = [0, 0, 0]
    for i in range(h):
        for j in range(w):
            for k in range(d):
                average[k] += image[i, j, k]
    for k in range(d):
        average[k] /= h*w

    for i in range(h):
        for j in range(w):
            for k in range(d):
                new_image[i, j, k] = average[k] + c * (image[i, j, k] - average[k])
                if new_image[i, j, k] > 255:
                    new_image[i, j, k] = 255
                if new_image[i, j, k] < 0:
                    new_image[i, j, k] = 0
    cv2.normalize(new_image, new_image, 0, 255, cv2.NORM_MINMAX)
    new_image = cv2.convertScaleAbs(new_image)
    return new_image


def contrast_trans_gray(c, image):
    h, w = image.shape
    new_image = numpy.zeros((h, w))
    average = 0
    for i in range(h):
        for j in range(w):
            average += image[i, j]
    average /= h*w

    for i in range(h):
        for j in range(w):
            new_image[i, j] = average + c * (image[i, j] - average)
            if new_image[i, j] > 255:
                new_image[i, j] = 255
            if new_image[i, j] < 0:
                new_image[i, j] = 0
    # cv2.normalize(new_image, new_image, 0, 255, cv2.NORM_MINMAX)
    new_image = cv2.convertScaleAbs(new_image)
    return new_image


def binarization_trans_gray(image):
    h, w = image.shape
    new_image = numpy.zeros((h, w))
    average = 0
    for i in range(h):
        for j in range(w):
            average += image[i, j]
    average /= h * w

    for i in range(h):
        for j in range(w):
            if image[i, j] < average:
                new_image[i, j] = 0
            else:
                new_image[i, j] = 255
    # cv2.normalize(new_image, new_image, 0, 255, cv2.NORM_MINMAX)
    new_image = cv2.convertScaleAbs(new_image)
    return new_image


def load_gif(gif_path):
    animation = pyglet.resource.animation(gif_path)
    sprite = pyglet.sprite.Sprite(animation)
    win = pyglet.window.Window(width=sprite.width, height=sprite.height)
    green = 0, 1, 0, 1
    pyglet.gl.glClearColor(*green)

    @win.event
    def on_draw():
        win.clear()
        sprite.draw()

    pyglet.app.run()


def function_1():
    image_path = "image/1_1.png"

    image_source = cv2.imread(image_path)
    cv2.imshow("source", image_source)

    # log_image = log_trans(0.2, image_source)
    # cv2.imshow("log image", log_image)

    # gamma_image = gamma_trans(1, 0.2, image_source)
    # cv2.imshow("gamma image", gamma_image)

    # contrast_image = contrast_trans(0.2, image_source)
    # cv2.imshow("contrast image", contrast_image)

    # image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度图方式读取
    # cv2.imshow("gray", image_gray)

    # log_image_gray = log_trans_gray(0.2, image_gray)
    # cv2.imshow("log image gray", log_image_gray)

    # gamma_image_gray = gamma_trans_gray(1, 1.0, image_gray)
    # cv2.imshow("game image gray", gamma_image_gray)

    # contrast_image_gray = contrast_trans_gray(0.5, image_gray)
    # cv2.imshow("contrast image gray", contrast_image_gray)

    # binarization_image_gray = binarization_trans_gray(image_gray)
    # cv2.imshow("binarization image gray", binarization_image_gray)

    # gif_path = "image/1_1.gif"
    # load_gif(gif_path)

    cv2.waitKey(0)
