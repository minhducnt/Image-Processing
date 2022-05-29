from matplotlib import pyplot as plt
import numpy as np
import cv2


# It is a method that improves the contrast in an image,
# in order to stretch out the intensity range


def make_histogram(image, bins=256):
    histogram = np.zeros(bins)
    for pixel in image:
        histogram[pixel] += 1
    return histogram


def normalize(entries):
    numerator = (entries - np.min(entries)) * 255
    denorminator = np.max(entries) - np.min(entries)
    result = numerator / denorminator
    result.astype("uint8")  # Convert float into int
    return result


def equalizehist(img):

    flatten_img = img.flatten()  # Convert array into 1D
    # cumsum() function returns the cumulative sum of the elements along the given axis.
    cumulative_sum = (make_histogram(flatten_img)).cumsum()
    cumulative_sum_norm = normalize(cumulative_sum)
    img_new_his = cumulative_sum_norm[flatten_img]
    # convert array back to original shape
    img_new = np.reshape(img_new_his, img.shape)
    return img_new, cumulative_sum_norm


def draw_image(orignal, result):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].imshow(orignal, cmap="gray")
    axes[0, 0].set_title("Result before equalization")
    axes[1, 0].hist(orignal.flatten(), 256, [0, 256])
    axes[0, 1].imshow(result, cmap="gray")
    axes[0, 1].set_title("Result after equalization")
    axes[1, 1].hist(result.ravel(), 256, [0, 256])
    fig.savefig("output/result.jpg")


img = cv2.imread("images/Alloy.jpg", 0)
result, normalized_cumsum = equalizehist(img)
draw_image(img, result)
