import cv2
import numpy as np

img = cv2.imread('harris_car.jpeg', cv2.IMREAD_GRAYSCALE)
img = cv2.GaussianBlur(img, (3, 3), 1)

Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

Ixx = cv2.GaussianBlur(Ix * Ix, (3, 3), 1)
Iyy = cv2.GaussianBlur(Iy * Iy, (3, 3), 1)
Ixy = cv2.GaussianBlur(Ix * Iy, (3, 3), 1)

k = 0.1
detH = Ixx * Iyy - Ixy ** 2
traceH = Ixx + Iyy
cH = detH - k * (traceH ** 2)

threshold = 0.01 * cH.max()
corner_mask = cH > threshold

local_max = (cH == cv2.dilate(cH, np.ones((7, 7), np.uint8)))

keypoints = np.argwhere(local_max & corner_mask)

img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for y, x in keypoints:
    cv2.circle(img_rgb, (x, y), 2, (0, 0, 255), -1)

cv2.imshow("Harris Corners", img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
