import cv2
import numpy as np
import argparse

""" Usage:
        cd <repo-folder>
        python image_warping.py [-h] --img IMG [--Width WIDTH] [--Height HEIGHT]
        
    Ex:
        python image_warping.py --img images/chocolate.jpg --Width 512 --Height 512
"""


def nothing(x):
    pass


def ParseArguments():
    parser = argparse.ArgumentParser(description="Perspective Image Warping Example")
    parser.add_argument(
        "--img", action="store", dest="img", help="Image Path", required=True
    )
    parser.add_argument(
        "--Width", action="store", dest="Width", type=int, help="Target Image Width"
    )
    parser.add_argument(
        "--Height", action="store", dest="Height", type=int, help="Target Image Height"
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = ParseArguments()
    org_img = cv2.imread(args.img)
    w = org_img.shape[1] if args.Width is None else args.Width
    h = org_img.shape[0] if args.Height is None else args.Height

    img = cv2.resize(org_img, (w, h))

    # Creating a window with a trackbar.
    WIN_WRAP_TIITLE = "Image Warp"
    WIN_RESIZED_WRAP_TIITLE = "Resized Image Warp"
    
    LEFT_WRAP = "Left Warp"
    RIGHT_WRAP = "Right Warp"
    TOP_WRAP = "Top Warp"
    BOTTOM_WRAP = "Bottom Warp"

    cv2.namedWindow(WIN_WRAP_TIITLE)
    Height, Width = img.shape[0:2]
    cv2.createTrackbar(LEFT_WRAP, WIN_WRAP_TIITLE, 0, 100, nothing)
    cv2.createTrackbar(RIGHT_WRAP, WIN_WRAP_TIITLE, 0, 100, nothing)
    cv2.createTrackbar(TOP_WRAP, WIN_WRAP_TIITLE, 0, 100, nothing)
    cv2.createTrackbar(BOTTOM_WRAP, WIN_WRAP_TIITLE, 0, 100, nothing)
    
    rect = np.array(
        [[0, 0], [Width - 1, 0], [Width - 1, Height - 1], [0, Height - 1]],
        dtype="float32",
    )

    warped = np.copy(img)
    while 1:
        cv2.imshow(WIN_WRAP_TIITLE, warped)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        left_wrap_factor = int(
            (cv2.getTrackbarPos(LEFT_WRAP, WIN_WRAP_TIITLE) / 100.0)
            * min(Width, Height)
        )
        right_wrap_factor = int(
            (cv2.getTrackbarPos(RIGHT_WRAP, WIN_WRAP_TIITLE) / 100.0)
            * min(Width, Height)
        )
        top_wrap_factor = int(
            (cv2.getTrackbarPos(TOP_WRAP, WIN_WRAP_TIITLE) / 100.0) * min(Width, Height)
        )
        bottom_wrap_factor = int(
            (cv2.getTrackbarPos(BOTTOM_WRAP, WIN_WRAP_TIITLE) / 100.0)
            * min(Width, Height)
        )

        dst = np.array(
            [
                [top_wrap_factor + left_wrap_factor, top_wrap_factor],
                [Width - (top_wrap_factor + right_wrap_factor + 1), top_wrap_factor],
                [
                    Width - (bottom_wrap_factor + right_wrap_factor + 1),
                    Height - (bottom_wrap_factor + right_wrap_factor + 1),
                ],
                [
                    left_wrap_factor + bottom_wrap_factor,
                    Height - (left_wrap_factor + bottom_wrap_factor + 1),
                ],
            ],
            dtype="float32",
        )
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (Width, Height))
        for i in range(4):
            cv2.circle(warped, (int(dst[i, 0]), int(dst[i, 1])), 5, (0, 255, 0), -1)

    cv2.destroyAllWindows()
