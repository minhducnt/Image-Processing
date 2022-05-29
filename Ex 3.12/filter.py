import cv2
import numpy as np

# Regular Sharpen
def sharpenImage(img):
    kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(img, -1, kernel_sharpening)

    return sharpened


# Edge Enhanced
def edgeEnhance(img):
    kernel_edge = (
        np.array(
            [
                [-1, -1, -1, -1, -1],
                [-1, 2, 2, 2, -1],
                [-1, 2, 8, 2, -1],
                [-2, 2, 2, 2, -1],
                [-1, -1, -1, -1, -1],
            ]
        )
        / 8.0
    )
    edge = cv2.filter2D(img, -1, kernel_edge)

    return edge


# Blur Image
def gaussianBlur(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    return blur


def medianBlur(img):
    median = cv2.medianBlur(img, 5)

    return median


# Noise Removal
def noiseRemoval(img):
    noiseRemoval = cv2.bilateralFilter(img, 5, 75, 75)

    return noiseRemoval


# Resize Image
def resizeImage(img):
    scale_percent = 60  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized
