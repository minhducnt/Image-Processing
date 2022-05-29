from matplotlib import pyplot as plt
import cv2
import numpy as np

# Getting the shape of the image.
img = cv2.imread("images/lena.jpg")
rows, cols, depth = img.shape

pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

# Transform matrix
M = cv2.getAffineTransform(pts1, pts2)
M = np.asmatrix(M)

# forward_warp_pic là kết quả của forward warped
# inverse_warp_pic là kết quả của inverse warped
forward_warp_pic = np.zeros_like(img)
inverse_warp_pic = np.zeros_like(img)

# Forward warping
# Đi qua img gốc, đưa tất cả các điểm transform trên img lên forward_warp_pic
for d in range(depth):
    for v in range(rows):
        for u in range(cols):
            x = int(round(M[0, 0] * u + M[0, 1] * v + M[0, 2]))
            y = int(round(M[1, 0] * u + M[1, 1] * v + M[1, 2]))
            if x < 0 or x >= cols or y < 0 or y >= rows:
                continue
            forward_warp_pic[y, x, d] = img[v, u, d]

# Inverse warping
inverse_warp_pic = cv2.warpAffine(img, M, (rows, cols))

# So sánh kết quả của hai Warp
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.plot(pts1[:, 0], pts1[:, 1], "ro")
plt.title("original Picture")

plt.subplot(2, 2, 2)
plt.imshow(forward_warp_pic)
plt.plot(pts2[:, 0], pts2[:, 1], "bo")
plt.title("Forward Warped Picture")

plt.subplot(2, 2, 3)
plt.imshow(img)
plt.plot(pts1[:, 0], pts1[:, 1], "ro")
plt.title("Original Picture")

plt.subplot(2, 2, 4)
plt.imshow(inverse_warp_pic)
plt.plot(pts2[:, 0], pts2[:, 1], "bo")
plt.title("Inverse Warped Picture")

plt.tight_layout()
plt.savefig(fname="results/forward_warp_vs_inverse_warp.png", dpi=100)
plt.show()
