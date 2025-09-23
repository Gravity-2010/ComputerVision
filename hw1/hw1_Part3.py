import cv2 # This imports OpenCV
import numpy as np # This imports NumPy
from matplotlib import pyplot as plt



# Task 2: Read and display an image
img = cv2.imread('elephant.jpeg') # Load the image
cv2.imshow("Elephant - OpenCV", img) # Display the image using OpenCV

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('elephant_copy.jpeg', img) # Save image to a file

plt.imshow(img)  
plt.title("Elephant - Matplotlib (wrong colors)")
plt.show()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title("Elephant - Matplotlib (correct colors)")
plt.show()

cv2.imwrite("elephant_matplotlib.png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))




# Task 3: Basic Image Processing OPerations

# Convert to grayscale
img_gray = cv2.imread("elephant.jpeg", cv2.IMREAD_GRAYSCALE)

cv2.imwrite("elephant_gray.png", img_gray)

plt.imshow(img_gray, cmap="gray")
plt.title("Grayscale Elephant")
plt.axis("off")
plt.show()


# Resize the image - Cropping the Baby Elephant
img_color = cv2.imread("elephant.jpeg")

img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

baby = img_rgb[350:910, 70:570]  

cv2.imwrite("elephant_baby.png", cv2.cvtColor(baby, cv2.COLOR_RGB2BGR))

plt.imshow(baby)
plt.title("Baby Elephant")
plt.axis("off")
plt.show()


# Resize the image - Downsampling
img_rgb = cv2.cvtColor(cv2.imread("elephant.jpeg"), cv2.COLOR_BGR2RGB)

height, width = img_rgb.shape[:2]
img_down = cv2.resize(img_rgb, (width // 10, height // 10))

cv2.imwrite("elephant_10xdown.png", cv2.cvtColor(img_down, cv2.COLOR_RGB2BGR))

plt.imshow(img_down)
plt.title("10x Downsampled Elephant")
plt.axis("off")
plt.show()


# Resize the image - Upsampling
# Nearest Neighbor
img_up_nearest = cv2.resize(img_down, (width, height), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("elephant_10xup_nearest.png", cv2.cvtColor(img_up_nearest, cv2.COLOR_RGB2BGR))

# Upsample with Bicubic
img_up_bicubic = cv2.resize(img_down, (width, height), interpolation=cv2.INTER_CUBIC)
cv2.imwrite("elephant_10xup_bicubic.png", cv2.cvtColor(img_up_bicubic, cv2.COLOR_RGB2BGR))

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img_up_nearest); axs[0].set_title("Nearest Neighbor")
axs[0].axis("off")
axs[1].imshow(img_up_bicubic); axs[1].set_title("Bicubic")
axs[1].axis("off")
plt.show()


# Abso;ute Difference
img_rgb = cv2.cvtColor(cv2.imread("elephant.jpeg"), cv2.COLOR_BGR2RGB)

diff_nearest = cv2.absdiff(img_rgb, img_up_nearest)
diff_bicubic = cv2.absdiff(img_rgb, img_up_bicubic)

cv2.imwrite("diff_nearest.png", cv2.cvtColor(diff_nearest, cv2.COLOR_RGB2BGR))
cv2.imwrite("diff_bicubic.png", cv2.cvtColor(diff_bicubic, cv2.COLOR_RGB2BGR))

error_nearest = diff_nearest.sum()
error_bicubic = diff_bicubic.sum()

print("Nearest Neighbor Error:", error_nearest)
print("Bicubic Error:", error_bicubic)




# task 4: Edge Detection and Image Blurring by Convolution
# Edge Detection with 2D Convolution
img = cv2.imread("elephant.jpeg", cv2.IMREAD_GRAYSCALE)

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

edges_x = cv2.filter2D(img, -1, sobel_x)

cv2.imwrite("edges_x.png", edges_x)


# Edge Detection using Laplacian Filter
laplacian_filter = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]])

edges_lap = cv2.filter2D(img, -1, laplacian_filter)

cv2.imwrite("edges_laplacian.png", edges_lap)


# Blurring with convolution
box_blur = np.ones((3,3), np.float32) / 9
blurred_box = cv2.filter2D(img, -1, box_blur)

cv2.imwrite("blurred_box.png", blurred_box)


# Gaussian Blur
gaussian_blur = cv2.getGaussianKernel(5, 1) @ cv2.getGaussianKernel(5, 1).T
blurred_gaussian = cv2.filter2D(img, -1, gaussian_blur)

cv2.imwrite("blurred_gaussian.png", blurred_gaussian)


# Display results
plt.figure(figsize=(12,6))

plt.subplot(2,3,1), plt.imshow(img, cmap='gray'), plt.title("Original")
plt.subplot(2,3,2), plt.imshow(edges_x, cmap='gray'), plt.title("Sobel Edge")
plt.subplot(2,3,3), plt.imshow(edges_lap, cmap='gray'), plt.title("Laplacian Edge")
plt.subplot(2,3,4), plt.imshow(blurred_box, cmap='gray'), plt.title("Box Blur")
plt.subplot(2,3,5), plt.imshow(blurred_gaussian, cmap='gray'), plt.title("Gaussian Blur")

plt.tight_layout()
plt.show()


# Convolution from Scratch
img = cv2.imread("elephant.jpeg", cv2.IMREAD_GRAYSCALE)
def convolve2d(image, kernel):

    kernel = [[kernel[i][j] for j in range(len(kernel[0])-1, -1, -1)] 
                             for i in range(len(kernel)-1, -1, -1)]
    
    h, w = len(image), len(image[0])
    kh, kw = len(kernel), len(kernel[0])
    pad_h, pad_w = kh // 2, kw // 2

    padded = [[0]*(w+2*pad_w) for _ in range(h+2*pad_h)]
    for i in range(h):
        for j in range(w):
            padded[i+pad_h][j+pad_w] = image[i][j]

    out = [[0]*w for _ in range(h)]

    for i in range(h):
        for j in range(w):
            val = 0
            for m in range(kh):
                for n in range(kw):
                    val += padded[i+m][j+n] * kernel[m][n]
            out[i][j] = min(max(int(val), 0), 255) 

    return np.array(out, dtype=np.uint8)

img_list = img.tolist()

sobel_y = [[-1,-2,-1],[0,0,0],[1,2,1]]
edges_manual = convolve2d(img_list, sobel_y)

box3 = [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]
blur_manual = convolve2d(img_list, box3)

cv2.imwrite("edges_manual.png", edges_manual)
cv2.imwrite("blur_manual.png", blur_manual)
