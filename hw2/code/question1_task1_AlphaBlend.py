import cv2
import numpy as np
import matplotlib.pyplot as plt

apple = cv2.imread('apple.jpeg')
orange = cv2.imread('orange.jpeg')

apple = cv2.cvtColor(apple, cv2.COLOR_BGR2RGB)
orange = cv2.cvtColor(orange, cv2.COLOR_BGR2RGB)

apple = cv2.resize(apple, (512, 512))
orange = cv2.resize(orange, (512, 512))

rows, cols, _ = apple.shape
mask = np.zeros((rows, cols), dtype=np.float32)
mask[:, :cols//2] = 0  
mask[:, cols//2:] = 1  

direct_blend = (1 - mask[..., None]) * apple + mask[..., None] * orange
direct_blend = direct_blend.astype(np.uint8)

blurred_mask = cv2.GaussianBlur(mask, (51, 51), 0)
alpha_blend = (1 - blurred_mask[..., None]) * apple + blurred_mask[..., None] * orange
alpha_blend = alpha_blend.astype(np.uint8)

fig, ax = plt.subplots(1, 4, figsize=(15, 6))
ax[0].imshow(apple); ax[0].set_title("Apple")
ax[1].imshow(orange); ax[1].set_title("Orange")
ax[2].imshow(direct_blend); ax[2].set_title("Direct Blend")
ax[3].imshow(alpha_blend); ax[3].set_title("Alpha Blend")
for a in ax: a.axis('off')
plt.show()

cv2.imwrite('apple_orange_direct.png', cv2.cvtColor(direct_blend, cv2.COLOR_RGB2BGR))
cv2.imwrite('apple_orange_alpha.png', cv2.cvtColor(alpha_blend, cv2.COLOR_RGB2BGR))
