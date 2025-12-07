import cv2
import numpy as np
import sys

def warp_and_composite_five(images, homographies, ref_idx=2):
    ref_img = images[ref_idx]
    h_ref, w_ref = ref_img.shape[:2]
    
    H12, H23, H34, H45 = homographies
    
    H13 = H23 @ H12
    H_to_ref = [
        H13,                              
        H23,                              
        np.eye(3),                        
        np.linalg.inv(H34),              
        np.linalg.inv(H34) @ np.linalg.inv(H45)  
    ]
    
    corners_list = []
    
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        
        if i == ref_idx:
            corners_transformed = corners
        else:
            corners_transformed = cv2.perspectiveTransform(corners, H_to_ref[i])
        
        corners_list.append(corners_transformed)
    
    all_corners = np.concatenate(corners_list, axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    translation = np.array([[1, 0, -x_min],
                           [0, 1, -y_min],
                           [0, 0, 1]], dtype=np.float32)
    
    mosaic_size = (x_max - x_min, y_max - y_min)
    mosaic = np.zeros((mosaic_size[1], mosaic_size[0], 3), dtype=np.uint8)
    
    warp_order = [0, 4, 1, 3, 2]  
    
    for idx in warp_order:
        H_final = translation @ H_to_ref[idx]
        img_warped = cv2.warpPerspective(images[idx], H_final, mosaic_size)
        
        mask = (img_warped != 0).any(axis=2)
        
        if idx == ref_idx:
            mosaic[mask] = img_warped[mask]
        else:
            mask_new = mask & ~(mosaic.any(axis=2))
            mosaic[mask_new] = img_warped[mask_new]
    
    return mosaic


def remove_black_borders(mosaic):
    gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return mosaic
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    cropped = mosaic[y:y+h, x:x+w]
    
    return cropped


if __name__ == "__main__":
    if len(sys.argv) != 11:
        print("Usage: python stitching_5.py <img1> <img2> <img3> <img4> <img5> <H12> <H23> <H34> <H45> <output>")
        print("Example: python stitching_5.py img1.jpg img2.jpg img3.jpg img4.jpg img5.jpg H12.npy H23.npy H34.npy H45.npy output.jpg")
        sys.exit(1)
    
    img_paths = sys.argv[1:6]
    H_paths = sys.argv[6:10]
    output_path = sys.argv[10]
    
    images = []
    for i, path in enumerate(img_paths, 1):
        img = cv2.imread(path)
        if img is None:
            print(f"Error: Could not load image {path}")
            sys.exit(1)
        images.append(img)
        print(f"  Image {i}: {img.shape[1]}x{img.shape[0]}")
    
    homographies = []
    for i, path in enumerate(H_paths, 1):
        H = np.load(path)
        homographies.append(H)
        print(f"  H{i}{i+1}: {path}")
    
    mosaic = warp_and_composite_five(images, homographies, ref_idx=2)
    print(f"Raw mosaic size: {mosaic.shape[1]}x{mosaic.shape[0]}")
    
    final_mosaic = remove_black_borders(mosaic)
    print(f"Final mosaic size: {final_mosaic.shape[1]}x{final_mosaic.shape[0]}")
    
    cv2.imwrite(output_path, final_mosaic)
    print(f"\nPanorama saved as '{output_path}'")