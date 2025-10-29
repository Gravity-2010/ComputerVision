import cv2
import numpy as np

def compute_harris_matrix(img, window_size=3, sigma=0.5):
    img = img.astype(np.float32)
    
    I_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    I_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    
    Ixx = I_x**2
    Iyy = I_y**2
    Ixy = I_x * I_y
    
    ksize = 5
    gaussian = cv2.getGaussianKernel(ksize, sigma)
    w = gaussian @ gaussian.T
    
    Sxx = cv2.filter2D(Ixx, -1, w, borderType=cv2.BORDER_REFLECT)
    Syy = cv2.filter2D(Iyy, -1, w, borderType=cv2.BORDER_REFLECT)
    Sxy = cv2.filter2D(Ixy, -1, w, borderType=cv2.BORDER_REFLECT)
    
    h, w_img = img.shape
    H_matrices = np.zeros((h, w_img, 2, 2), dtype=np.float32)
    H_matrices[:, :, 0, 0] = Sxx
    H_matrices[:, :, 0, 1] = Sxy
    H_matrices[:, :, 1, 0] = Sxy
    H_matrices[:, :, 1, 1] = Syy
    
    return H_matrices, (Sxx, Sxy, Syy)

img = cv2.imread('harris_sunflower.jpg', cv2.IMREAD_GRAYSCALE)
H, (Sxx, Sxy, Syy) = compute_harris_matrix(img, window_size=5, sigma=0.5)
print("H matrix at pixel (10,10):\n", H[10, 10])

def compute_corner_strength(Sxx, Sxy, Syy, k=0.1):
    detH = (Sxx * Syy) - (Sxy ** 2)
    traceH = Sxx + Syy
    
    cH = detH - k * (traceH ** 2)
    return cH

# Task 2: Compute corner strength
corner_strength = compute_corner_strength(Sxx, Sxy, Syy, k=0.1)

print("Corner strength at pixel (10,10):", corner_strength[10, 10])
