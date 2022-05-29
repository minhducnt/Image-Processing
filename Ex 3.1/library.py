import cv2
import numpy as np

# Hàm chỉnh độ sáng cho ảnh
def gammaCorrection(src, gamma):
    """
    It takes an image and a percentage, and returns the image with the percentage of pixels in the image
    that are the darkest and the percentage of pixels in the image that are the brightest set to black
    and white respectively

    :param src: The source image
    :param gamma: The gamma value to be used in the gamma correction
    :return: a list of the three channels of the image.
    """
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


# Hàm chỉnh màu cho ảnh
def colorBalance(img, percent=1):
    """
    It takes an image and a percentage, and returns the image with the percentage of pixels in the image
    that are the darkest and the percentage of pixels in the image that are the brightest set to black
    and white respectively

    :param img: The image to be processed
    :param percent: The percentage of the image to be clipped, defaults to 1 (optional)
    :return: A list of the three channels of the image.
    """
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0),
    )
    for channel in cv2.split(img):
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0, 256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate(
            (
                np.zeros(low_cut),
                np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
                255 * np.ones(255 - high_cut),
            )
        )
        out_channels.append(cv2.LUT(channel, lut.astype("uint8")))
    return cv2.merge(out_channels)
