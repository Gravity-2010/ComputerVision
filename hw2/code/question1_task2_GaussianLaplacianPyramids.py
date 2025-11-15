import cv2
import numpy as np
import matplotlib.pyplot as plt

def build_gaussian_pyramid(img, levels=5):
    gaussian_pyramid = [img.astype(np.float32)]
    for i in range(1, levels):
        blurred = cv2.GaussianBlur(gaussian_pyramid[-1], (5, 5), 0)
        down = blurred[::2, ::2] 
        gaussian_pyramid.append(down)
    return gaussian_pyramid

def build_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        g_curr = gaussian_pyramid[i]
        g_next = gaussian_pyramid[i + 1]
        g_next_up = cv2.resize(g_next, (g_curr.shape[1], g_curr.shape[0]), interpolation=cv2.INTER_LINEAR)
        laplacian = g_curr - g_next_up
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid

def reconstruct_from_laplacian(laplacian_pyramid):
    img = laplacian_pyramid[-1]
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        upsampled = cv2.resize(img, (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0]), interpolation=cv2.INTER_LINEAR)
        img = upsampled + laplacian_pyramid[i]
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


if __name__ == "__main__":
    img = cv2.imread("apple.jpeg")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    gaussian_pyramid = build_gaussian_pyramid(img_rgb, levels=5)
    laplacian_pyramid = build_laplacian_pyramid(gaussian_pyramid)
    
    reconstructed = reconstruct_from_laplacian(laplacian_pyramid)
    
    error = np.mean(np.abs(img_rgb.astype(np.float32) - reconstructed.astype(np.float32)))
    print(f'Reconstruction error (mean absolute difference): {error:.4f}')
 
    plt.figure(figsize=(12, 8))

    for i, g in enumerate(gaussian_pyramid):
        plt.subplot(2, len(gaussian_pyramid), i + 1)
        plt.imshow(g.astype(np.uint8))
        plt.title(f'Gaussian {i}')
        plt.axis('off')

    for i, l in enumerate(laplacian_pyramid):
        plt.subplot(2, len(laplacian_pyramid), len(gaussian_pyramid) + i + 1)
        plt.imshow(np.clip(l - l.min(), 0, 255).astype(np.uint8))
        plt.title(f'Laplacian {i}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    cv2.imwrite("reconstructed_pyramid.png", cv2.cvtColor(reconstructed, cv2.COLOR_RGB2BGR))

# unit test whether your pyramid code is accurate.
img = cv2.imread("apple.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

g1_cv = cv2.pyrDown(img)
g2_cv = cv2.pyrDown(g1_cv)

g_pyramid = build_gaussian_pyramid(img, 3)
diff = np.mean((g_pyramid[1] - g1_cv) ** 2)
print(f"Difference between custom and cv2.pyrDown: {diff:.6f}")
