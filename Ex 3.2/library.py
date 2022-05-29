import cv2
import numpy as np

# Hàm lấy kích thước của ảnh
def get_shape(img):
    """
    It takes an image as input and returns the width and height of the image

    :param img: The image to be processed
    :return: The width and height of the image.
    """
    height = img.shape[1]
    width = img.shape[0]

    return width, height


# Hàm tạo kernel
def get_kernel():
    """
    It returns a tuple of three 5x5 numpy arrays, each with a value of 1/6 in the first four positions
    and 1/12 in the center position
    :return: the kernel for each channel.
    """
    b_kernel = np.ones((5, 5), np.uint8) / 6
    g_kernel = np.ones((5, 5), np.uint8) / 6
    r_kernel = np.ones((5, 5), np.uint8) / 12

    # b_kernel = np.array([[1.5, 0, 1.5], [0, 0, 0], [1.5, 0, 1.5]]) / 4
    # g_kernel = np.array([[0, 0, 0], [0, 2.5, 0], [0, 0, 0]]) / 2
    # r_kernel = np.array([[0, 1.5, 0], [1.5, 0, 1.5], [0, 1.5, 0]]) / 5

    # b_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) / 1
    # g_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) / 4
    # r_kernel = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]) / 1

    # b_kernel = np.array([[2, 0, 2], [0, 0, 0], [2, 0, 2]]) / 5
    # g_kernel = np.array([[0, 0, 0], [0, 4, 0], [0, 0, 0]]) / 2
    # r_kernel = np.array([[0, 2, 0], [2, 0, 2], [0, 2, 0]]) / 8

    return b_kernel, g_kernel, r_kernel


# Hàm tạo mask
def fetch_mask(shape):
    """
    It creates a mask for each color channel, where the mask is a 2D array of the same size as the
    image, and each element of the mask is either 0 or 1

    :param shape: The shape of the image
    :return: three masks, one for each color channel.
    """
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


# Hàm chuẩn bị channel
def prepare_channel(channel_index, channel_matrix, shape):
    """
    It takes a channel index, a channel matrix, and a shape, and returns the channel and a filtered
    version of the channel

    :param channel_index: The index of the channel we're working on
    :param channel_matrix: The image matrix
    :param shape: The shape of the image
    :return: The channel and filtered are being returned.
    """
    channel = cv2.bitwise_and(
        channel_matrix, channel_matrix, mask=fetch_mask(shape)[channel_index]
    )
    filtered = cv2.filter2D(channel, -1, get_kernel()[channel_index])

    return channel, filtered


def bill_freeman(blue, green, red):
    g_r = green - red
    b_r = blue - red

    g_r = cv2.medianBlur(g_r, 1)
    b_r = cv2.medianBlur(b_r, 1)

    g_r = g_r + red
    b_r = b_r + red

    final_img = cv2.merge((b_r, g_r, red))
    return final_img


def display_squared_values(b_org, g_org, r_org, blue, green, red):
    b_sq = calculate_squared(b_org, blue)
    g_sq = calculate_squared(g_org, green)
    r_sq = calculate_squared(r_org, red)

    square_image = np.array(b_sq + g_sq + r_sq).astype(np.uint8)
    return square_image


def calculate_squared(org, out):
    return np.sqrt(np.square(org) - np.square(out))