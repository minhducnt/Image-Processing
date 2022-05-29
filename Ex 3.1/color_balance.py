import cv2
import numpy as np
import library as lib


if __name__ == "__main__":
    # Reading the image from the path and storing it in the variable img.
    img = cv2.imread("images/ryan.jpg")

    # Applying color balance to the image.
    # out = lib.colorBalance(img, 1)
    # res1 = np.hstack((img, out))
    # cv2.imshow("Before vs After", res1)

    # # Applying gamma correction to the image.
    gammaImg = lib.gammaCorrection(img, 1)
    res2 = np.hstack((img, gammaImg))
    cv2.imshow("Original image vs Gamma correction", res2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
