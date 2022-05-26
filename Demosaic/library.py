import cv2
import numpy as np


def get_shape(img):
    height = img.shape[1]
    width = img.shape[0]

    return width, height


def get_kernel():
    b_kernel = np.ones((5, 5), np.uint8) / 6
    g_kernel = np.ones((5, 5), np.uint8) / 6
    r_kernel = np.ones((5, 5), np.uint8) / 12

    return b_kernel, g_kernel, r_kernel


def fetch_mask(shape):
    b_mask = np.zeros(shape, dtype=np.uint8)
    b_mask[:, ::2] = 1
    b_mask[1::2] = 0

    g_mask = np.zeros(shape, dtype=np.uint8)
    g_mask[1::2, 1::2] = 1

    r_mask = np.zeros(shape, dtype=np.uint8)
    r_mask[:, 1::2] = 1
    r_mask[1::2] = 0
    r_mask[1::2, ::2] = 1

    return b_mask, g_mask, r_mask


def prepare_channel(channel_index, channel_matrix, shape):
    channel = cv2.bitwise_and(
        channel_matrix, channel_matrix, mask=fetch_mask(shape)[channel_index]
    )
    filtered = cv2.filter2D(channel, -1, get_kernel()[channel_index])

    return channel, filtered
