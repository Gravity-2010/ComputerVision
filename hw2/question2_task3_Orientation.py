import numpy as np
import cv2

def compute_orientation(Ix, Iy, degrees=False):
    orientation = np.arctan2(Iy, Ix) 
    if degrees:
        orientation = np.degrees(orientation)
    return orientation

img = cv2.imread('harris_elephant.jpeg', cv2.IMREAD_GRAYSCALE)
Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
orientation = compute_orientation(Ix, Iy, degrees=True)

print("Orientation at (100, 100):", orientation[100, 100], "degrees")
