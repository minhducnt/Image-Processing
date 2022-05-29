import numpy as np
import cv2 as cv
import library as lib

if __name__ == "__main__":
    compA = cv.imread("images/flowers-compA.jpg") / 255.0
    compB = cv.imread("images/flowers-compB.jpg") / 255.0
    backA = cv.imread("images/flowers-backA.jpg") / 255.0
    backB = cv.imread("images/flowers-backB.jpg") / 255.0
    realBack = cv.imread("images/window.jpg") / 255.0

    color, alpha = lib.trigulationMatting(compA, compB, backA, backB)
    color = color * 255.0

    cv.imwrite("flower-alpha4.jpg", alpha * 255.0)
    cv.imwrite("flower-foreground4.jpg", color)

    back = np.zeros(compA.shape)
    newAlpha = np.zeros(backA.shape)
    newAlpha[:, :, 0] = alpha
    newAlpha[:, :, 1] = alpha
    newAlpha[:, :, 2] = alpha
    back = realBack * (1.0 - newAlpha)
    back = back * 255.0
    cv.imwrite("flower-back4.jpg", back)
    composite = color + back
    cv.imwrite("flower-composite4.jpg", composite)
