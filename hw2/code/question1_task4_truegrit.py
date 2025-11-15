import cv2
import numpy as np
import matplotlib.pyplot as plt
from question1_task3_MultiresolutionBlending import multiblend 

def direct_blend(img1, img2, mask):
    return ((1 - mask[..., None]) * img1 + mask[..., None] * img2).astype(np.uint8)

def alpha_blend(img1, img2, mask, blur_ksize=51):
    blurred = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
    return ((1 - blurred[..., None]) * img1 + blurred[..., None] * img2).astype(np.uint8)

mascot = cv2.imread('mascot.png')
real_dog = cv2.imread('realdog.png')
mascot = cv2.cvtColor(cv2.resize(mascot, (512, 512)), cv2.COLOR_BGR2RGB)
real_dog = cv2.cvtColor(cv2.resize(real_dog, (512, 512)), cv2.COLOR_BGR2RGB)

rows, cols = mascot.shape[:2]
mask = np.zeros((rows, cols), dtype=np.float32)
mask[:, :cols//2] = 0   
mask[:, cols//2:] = 1   

direct = direct_blend(mascot, real_dog, mask)
alpha = alpha_blend(mascot, real_dog, mask)
laplacian = multiblend(mascot, real_dog, mask, levels=5)

fig, ax = plt.subplots(1, 4, figsize=(20,5))
ax[0].imshow(mascot); ax[0].set_title("Mascot")
ax[1].imshow(real_dog); ax[1].set_title("Real Dog")
ax[2].imshow(direct); ax[2].set_title("Direct Blend")
ax[3].imshow(alpha); ax[3].set_title("Alpha Blend")
for a in ax: a.axis("off")
plt.show()

plt.figure(figsize=(6,6))
plt.imshow(laplacian); plt.title("Laplacian Blend"); plt.axis("off")
plt.show()
