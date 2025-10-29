import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom

def gaussian_pyramid(image, levels=5, sigma=1):
    g_pyramid = [image.astype(np.float32)]
    for _ in range(1, levels):
        blurred = gaussian_filter(g_pyramid[-1], sigma=(sigma, sigma, 0))
        downsampled = blurred[::2, ::2]
        g_pyramid.append(downsampled)
    return g_pyramid

def laplacian_pyramid(g_pyr):
    l_pyr = []
    for i in range(len(g_pyr) - 1):
        upsampled = zoom(g_pyr[i + 1], (2, 2, 1), order=1)
        upsampled = upsampled[:g_pyr[i].shape[0], :g_pyr[i].shape[1], :]
        l_pyr.append(g_pyr[i] - upsampled)
    l_pyr.append(g_pyr[-1])
    return l_pyr

def reconstruct_from_laplacian(l_pyr):
    image = l_pyr[-1]
    for i in range(len(l_pyr) - 2, -1, -1):
        upsampled = zoom(image, (2, 2, 1), order=1)
        upsampled = upsampled[:l_pyr[i].shape[0], :l_pyr[i].shape[1], :]
        image = upsampled + l_pyr[i]
    return np.clip(image, 0, 255).astype(np.uint8)

def multiblend(img1, img2, mask, levels=5):
    h, w = img1.shape[:2]
    img2 = cv2.resize(img2, (w, h))
    mask = cv2.resize(mask, (w, h))

    if len(mask.shape) == 2:
        mask = cv2.merge([mask, mask, mask])
    mask = mask.astype(np.float32) / 255.0

    gp_mask = gaussian_pyramid(mask, levels)
    lp_img1 = laplacian_pyramid(gaussian_pyramid(img1, levels))
    lp_img2 = laplacian_pyramid(gaussian_pyramid(img2, levels))

    lp_blend = []
    for L1, L2, GM in zip(lp_img1, lp_img2, gp_mask):
        blended = GM * L2 + (1 - GM) * L1
        lp_blend.append(blended)

    blended_image = reconstruct_from_laplacian(lp_blend)
    return blended_image

if __name__ == "__main__":
    img1 = cv2.imread('apple.jpeg')
    img2 = cv2.imread('orange.jpeg')

    img3 = cv2.imread('mascot.png')
    img4 = cv2.imread('realdog.png')

    mask = np.zeros_like(img1[:, :, 0], dtype=np.uint8)
    mask[:, :mask.shape[1] // 2] = 255

    blended = multiblend(img1, img2, mask, levels=5)

    blended2 = multiblend(img3, img4, mask, levels=5)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1); plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)); plt.title("Image 1 (Apple)"); plt.axis("off")
    plt.subplot(1, 4, 2); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)); plt.title("Image 2 (Orange)"); plt.axis("off")
    plt.subplot(1, 4, 3); plt.imshow(mask, cmap="gray"); plt.title("Mask"); plt.axis("off")
    plt.subplot(1, 4, 4); plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)); plt.title("Blended (Orapple)"); plt.axis("off")
    plt.tight_layout()

# Task 4
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1); plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)); plt.title("Image 3 (Mascot)"); plt.axis("off")
    plt.subplot(1, 4, 2); plt.imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)); plt.title("Image 4 (Real Dog)"); plt.axis("off")
    plt.subplot(1, 4, 3); plt.imshow(mask, cmap="gray"); plt.title("Mask"); plt.axis("off")
    plt.subplot(1, 4, 4); plt.imshow(cv2.cvtColor(blended2, cv2.COLOR_BGR2RGB)); plt.title("Blended"); plt.axis("off")
    plt.tight_layout()

    plt.show()
