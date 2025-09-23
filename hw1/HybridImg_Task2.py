import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread("elephant_baby.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("tiger.jpeg", cv2.IMREAD_GRAYSCALE)

img1 = cv2.resize(img1, (512, 512))
img2 = cv2.resize(img2, (512, 512))

low_pass = cv2.GaussianBlur(img1, (9, 9), 10)

blurred_img2 = cv2.GaussianBlur(img2, (15, 15), 10)
high_pass = cv2.subtract(img2, blurred_img2)

low_pass = cv2.normalize(low_pass, None, 0, 255, cv2.NORM_MINMAX)
high_pass = cv2.normalize(high_pass, None, 0, 255, cv2.NORM_MINMAX)

hybrid = cv2.addWeighted(low_pass, 0.8, high_pass, 0.2, 0)

plt.figure(figsize=(15,5))

plt.subplot(1,4,1)
plt.imshow(img1, cmap='gray')
plt.title("Original Image 1")

plt.subplot(1,4,2)
plt.imshow(img2, cmap='gray')
plt.title("Original Image 2")

plt.subplot(1,4,3)
plt.imshow(low_pass, cmap='gray')
plt.title("Low-pass Filtered")

plt.subplot(1,4,4)
plt.imshow(high_pass, cmap='gray')
plt.title("High-pass Filtered")

plt.figure(figsize=(6,6))
plt.imshow(hybrid, cmap='gray')
plt.title("Hybrid Image")
plt.show()
