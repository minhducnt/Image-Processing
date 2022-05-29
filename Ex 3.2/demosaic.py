import cv2
import library as lib
import numpy as np

from library import prepare_channel, display_squared_values, bill_freeman, get_shape


def main():
    # Reading the image from the path and storing it in the variable img.
    img = cv2.imread("images/crayons_mosaic.bmp")
    org = cv2.imread("images/crayons.jpg")

    # Splitting the image into three channels, blue, green and red.
    b, g, r = cv2.split(img)
    b_o, g_o, r_o = cv2.split(org)

    # Getting the shape of the image.
    shape = get_shape(img)

    # Preparing the channels for the final image.
    b_org, blue = prepare_channel(0, b, shape)
    g_org, green = prepare_channel(1, g, shape)
    r_org, red = prepare_channel(2, r, shape)

    # Merging the three channels into one image.
    final_img = cv2.merge((blue, green, red))
    squared_img = display_squared_values(b_o, g_o, r_o, blue, green, red)
    freeman_img = bill_freeman(blue, green, red)

    # Concatenating the two images horizontally.
    numpy_horizontal_concat = np.concatenate((img, final_img), axis=1)

    # Displaying the images.
    cv2.imshow("Photo Demosaic", numpy_horizontal_concat)
    # cv2.imshow("Root Squared", squared_img)
    # cv2.imshow("Freeman", freeman_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
