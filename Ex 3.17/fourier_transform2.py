#%% Importing the libraries that we need to use.
import cv2
import numpy as np
import matplotlib.pyplot as plt

#%% Reading the image and converting it to grayscale.
woman = cv2.imread("images/woman.png", 0)
square = cv2.imread("images/square.png", 0)

#%% Plotting the image Woman.
plt.subplot(331), plt.imshow(woman, "gray"), plt.title("Woman")
plt.xticks([]), plt.yticks([])

#%% Take its amplitude and phase
f1 = np.fft.fft2(woman)
f1shift = np.fft.fftshift(
    f1
)  # Move the low-frequency part to the middle for easy observation Move the low-frequency part to the middle for easy observation
f1_A = np.abs(f1shift)  # Amplitude
f1_P = np.angle(f1shift)  # Phase

#%% Displays its amplitude and phase
plt.subplot(332), plt.imshow(f1_A, "gray"), plt.title("Woman_Magnitude")
plt.subplot(333), plt.imshow(f1_P, "gray"), plt.title("Woman_Phase")

#%% Plotting the image Square
plt.subplot(334), plt.imshow(square, "gray"), plt.title("Square")
plt.xticks([]), plt.yticks([])

#%% Take its amplitude and phase
f2 = np.fft.fft2(square)
f2shift = np.fft.fftshift(
    f2
)  # Move the low-frequency part to the middle for easy observation Move the low-frequency part to the middle for easy observation
f2_A = np.abs(f2shift)  # amplitude
f2_P = np.angle(f2shift)  # Phase

#%% Displays its amplitude and phase
plt.subplot(335), plt.imshow(f2_A, "gray"), plt.title("Square_Magnitude")
plt.subplot(336), plt.imshow(f2_P, "gray"), plt.title("Square_Phase")

#%% Woman's amplitude - Square's phase
img_new1_f = np.zeros(woman.shape, dtype=complex)
img1_real = f1_A * np.cos(f2_P)  # Take the real part
img1_imag = f1_A * np.sin(f2_P)  # Take the imaginary part
img_new1_f.real = np.array(img1_real)
img_new1_f.imag = np.array(img1_imag)
f3shift = np.fft.ifftshift(img_new1_f)  # Fourier inverse conversion
img_new1 = np.fft.ifft2(f3shift)

#%% The output result img_new1 complex and cannot be displayed yet
img_new1 = np.abs(img_new1)
plt.subplot(337), plt.imshow(img_new1, "gray"), plt.title("Woman_A + Square_P")
plt.xticks([]), plt.yticks([])

#%% Square's amplitude - Woman's phase
img_new2_f = np.zeros(woman.shape, dtype=complex)
img2_real = f2_A * np.cos(f1_P)  # Take the real part
img2_imag = f2_A * np.sin(f1_P)  # Take the imaginary part
img_new2_f.real = np.array(img2_real)
img_new2_f.imag = np.array(img2_imag)
f4shift = np.fft.ifftshift(img_new2_f)  # Fourier inverse conversion
img_new2 = np.fft.ifft2(f4shift)

#%% The output result img_new2 a complex number and cannot be displayed yet
img_new2 = np.abs(img_new2)
plt.subplot(338), plt.imshow(img_new2, "gray"), plt.title("Woman_P + Square_A")
plt.xticks([]), plt.yticks([])
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
