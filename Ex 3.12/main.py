import cv2
import numpy as np
import filter as lib

if __name__ == "__main__":
    # The path to the image that we want to sharpen.
    path = "images\cat.png"

    # Reading the image from the path and storing it in the variable img.
    img = cv2.imread(path)

    # Resize the image to a smaller size.
    img = lib.resizeImage(img)

    # Regular Sharpen
    sharp = lib.sharpenImage(img)

    # Edge Enhanced
    edge = lib.edgeEnhance(img)

    # Blur Image
    blur = lib.medianBlur(img)

    # Noise Removal
    noise = lib.noiseRemoval(img)

    # Stacking the images horizontally.
    res = np.hstack((img, sharp, edge, blur, noise))

    # Display the images.
    cv2.imshow(
        "Normal vs Sharpening Image vs Edge Enhanced vs Bluring Image vs Noise Removal",
        res,
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()
