import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, convolve, maximum_filter
import cv2

# Task 1: H-matrix compuation
def task1_compute_H(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    Ix = convolve(image.astype(float), sobel_x, mode='reflect')
    Iy = convolve(image.astype(float), sobel_y, mode='reflect')
    
    Sx2 = gaussian_filter(Ix**2, sigma=0.5, mode='reflect')
    Sy2 = gaussian_filter(Iy**2, sigma=0.5, mode='reflect')
    Sxy = gaussian_filter(Ix*Iy, sigma=0.5, mode='reflect')
    
    return Sx2, Sxy, Sy2, Ix, Iy

# Task 2: Corner Strength Function
def task2_corner_strength(Sx2, Sxy, Sy2):
    det_H = Sx2 * Sy2 - Sxy**2
    trace_H = Sx2 + Sy2
    c = det_H - 0.1 * (trace_H**2)
    return c

# Task 3: Orientation at each pixel
def task3_orientation(Ix, Iy):
    return np.arctan2(Iy, Ix)

# Task 4: Threshold
def task4_select_keypoints(c):
    threshold = np.percentile(c, 99)
    local_max = maximum_filter(c, size=5)
    keypoints = np.argwhere((c == local_max) & (c > threshold))
    return keypoints

# Task 5: Display keypoints
def task5_display(image, keypoints, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')
    for kp in keypoints:
        circle = plt.Circle((kp[1], kp[0]), 3, color='red', fill=False, linewidth=1.5)
        plt.gca().add_patch(circle)
    plt.title(f'{title} ({len(keypoints)} keypoints)')
    plt.axis('off')

# Task 6: Different corner strength function (Nobel 1998)
def task6_alternative_strength(Sx2, Sxy, Sy2, epsilon=1e-4):
    det_H = Sx2 * Sy2 - Sxy**2
    trace_H = Sx2 + Sy2
    c = det_H / (trace_H + epsilon)
    return c

# Task 7: Compare provided and custom images
def task7_compare(provided_img, custom_img, object_name):
    Sx2, Sxy, Sy2, Ix, Iy = task1_compute_H(provided_img)
    c = task2_corner_strength(Sx2, Sxy, Sy2)
    kp_provided = task4_select_keypoints(c)
    
    Sx2, Sxy, Sy2, Ix, Iy = task1_compute_H(custom_img)
    c = task2_corner_strength(Sx2, Sxy, Sy2)
    kp_custom = task4_select_keypoints(c)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(provided_img, cmap='gray')
    for kp in kp_provided:
        circle = plt.Circle((kp[1], kp[0]), 3, color='red', fill=False, linewidth=1.5)
        axes[0].add_patch(circle)
    axes[0].set_title(f'Provided {object_name} ({len(kp_provided)} keypoints)')
    axes[0].axis('off')
    
    axes[1].imshow(custom_img, cmap='gray')
    for kp in kp_custom:
        circle = plt.Circle((kp[1], kp[0]), 3, color='green', fill=False, linewidth=1.5)
        axes[1].add_patch(circle)
    axes[1].set_title(f'Custom {object_name} ({len(kp_custom)} keypoints)')
    axes[1].axis('off')
    
    plt.suptitle(f'Task 7: {object_name}')
    
    print(f"{object_name} - Provided: {len(kp_provided)} keypoints, Custom: {len(kp_custom)} keypoints")

def main():
    images = ['harris_car.jpeg', 'harris_elephant.jpeg', 'harris_sunflower.jpg']
    names = ['Car', 'Elephant', 'Sunflower']
    
    for img_file, name in zip(images, names):
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        Sx2, Sxy, Sy2, Ix, Iy = task1_compute_H(img)
        c = task2_corner_strength(Sx2, Sxy, Sy2)
        orientation = task3_orientation(Ix, Iy)
        keypoints = task4_select_keypoints(c)
        task5_display(img, keypoints, name)
        print(f"{name}: {len(keypoints)} keypoints detected")
    
    for img_file, name in zip(images, names):
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        Sx2, Sxy, Sy2, Ix, Iy = task1_compute_H(img)
        
        c_harris = task2_corner_strength(Sx2, Sxy, Sy2)
        kp_harris = task4_select_keypoints(c_harris)
        
        c_alt = task6_alternative_strength(Sx2, Sxy, Sy2)
        kp_alt = task4_select_keypoints(c_alt)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].imshow(img, cmap='gray')
        for kp in kp_harris:
            circle = plt.Circle((kp[1], kp[0]), 3, color='red', fill=False, linewidth=1.5)
            axes[0].add_patch(circle)
        axes[0].set_title(f'Harris: det(H)-0.1*tr(H)² ({len(kp_harris)} keypoints)')
        axes[0].axis('off')
        
        axes[1].imshow(img, cmap='gray')
        for kp in kp_alt:
            circle = plt.Circle((kp[1], kp[0]), 3, color='blue', fill=False, linewidth=1.5)
            axes[1].add_patch(circle)
        axes[1].set_title(f'Nobel (1998): det(H)/(tr(H)+ε) ({len(kp_alt)} keypoints)')
        axes[1].axis('off')
        
        plt.suptitle(f'Task 6: {name}')
        
        print(f"{name} - Harris: {len(kp_harris)} keypoints, Alt: {len(kp_alt)} keypoints")
    
    # Task 7: Compare provided vs custom images
    custom_images = ['my_car.png', 'my_elephant.png', 'my_sunflower.png']
    
    for provided_file, custom_file, name in zip(images, custom_images, names):
        provided = cv2.imread(provided_file, cv2.IMREAD_GRAYSCALE)
        custom = cv2.imread(custom_file, cv2.IMREAD_GRAYSCALE)
        
        if provided is not None and custom is not None:
            task7_compare(provided, custom, name)
    
    plt.show()

if __name__ == "__main__":
    main()