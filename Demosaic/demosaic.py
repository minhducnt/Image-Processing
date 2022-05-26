import cv2
import library as lb
import numpy as np

if __name__ == "__main__":
    img = cv2.imread("images/crayons_mosaic.bmp")
    org = cv2.imread("images/crayons.jpg.jpg")

    b, g, r = cv2.split(img)

    shape = lb.get_shape(img)

    blue = lb.prepare_channel(0, b, shape)
    green = lb.prepare_channel(1, g, shape)
    red = lb.prepare_channel(2, r, shape)

    final_img = cv2.merge((blue, green, red))

    numpy_horizontal_concat = np.concatenate((img, final_img), axis=1)
    cv2.imshow("Photo Demosaic", numpy_horizontal_concat)
    cv2.waitKey()
