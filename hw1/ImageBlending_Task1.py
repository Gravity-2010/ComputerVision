import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread("apple.jpeg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("orange.jpeg", cv2.IMREAD_GRAYSCALE)

h, w = img1.shape
img2 = cv2.resize(img2, (w, h))

f1 = np.fft.fft2(img1)
f2 = np.fft.fft2(img2)

mag1, phase1 = np.abs(f1), np.angle(f1)
mag2, phase2 = np.abs(f2), np.angle(f2)

f1_new = mag1 * np.exp(1j * phase2)  # img1 mag + img2 phase
f2_new = mag2 * np.exp(1j * phase1)  # img2 mag + img1 phase

img1_recon = np.fft.ifft2(f1_new).real
img2_recon = np.fft.ifft2(f2_new).real

plt.figure(figsize=(12,8))

plt.subplot(2,3,1), plt.imshow(img1, cmap='gray'), plt.title("Original Image 1")
plt.subplot(2,3,2), plt.imshow(np.log1p(mag1), cmap='gray'), plt.title("Magnitude 1")
plt.subplot(2,3,3), plt.imshow(phase1, cmap='gray'), plt.title("Phase 1")

plt.subplot(2,3,4), plt.imshow(img2, cmap='gray'), plt.title("Original Image 2")
plt.subplot(2,3,5), plt.imshow(np.log1p(mag2), cmap='gray'), plt.title("Magnitude 2")
plt.subplot(2,3,6), plt.imshow(phase2, cmap='gray'), plt.title("Phase 2")

plt.show()

plt.figure(figsize=(10,5))
plt.subplot(1,2,1), plt.imshow(img1_recon, cmap='gray'), plt.title("Img1 Mag + Img2 Phase")
plt.subplot(1,2,2), plt.imshow(img2_recon, cmap='gray'), plt.title("Img2 Mag + Img1 Phase")
plt.show()
