import cv2
import numpy as np
import sys

def warp_and_composite_multiple(images, homographies, ref_idx=1):
    ref_img = images[ref_idx]
    h_ref, w_ref = ref_img.shape[:2]
    
    H12, H23 = homographies
    H23_inv = np.linalg.inv(H23)
    
    corners_list = []
    
    corners_ref = np.float32([[0, 0], [0, h_ref], [w_ref, h_ref], [w_ref, 0]]).reshape(-1, 1, 2)
    corners_list.append(corners_ref)
    
    h1, w1 = images[0].shape[:2]
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners1_transformed = cv2.perspectiveTransform(corners1, H12)
    corners_list.append(corners1_transformed)
    
    h3, w3 = images[2].shape[:2]
    corners3 = np.float32([[0, 0], [0, h3], [w3, h3], [w3, 0]]).reshape(-1, 1, 2)
    corners3_transformed = cv2.perspectiveTransform(corners3, H23_inv)
    corners_list.append(corners3_transformed)
    
    all_corners = np.concatenate(corners_list, axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    translation = np.array([[1, 0, -x_min],
                           [0, 1, -y_min],
                           [0, 0, 1]], dtype=np.float32)
    
    mosaic_size = (x_max - x_min, y_max - y_min)
    mosaic = np.zeros((mosaic_size[1], mosaic_size[0], 3), dtype=np.uint8)
    
    img1_warped = cv2.warpPerspective(images[0], translation @ H12, mosaic_size)
    mask1 = (img1_warped != 0).any(axis=2)
    mosaic[mask1] = img1_warped[mask1]
    
    img2_warped = cv2.warpPerspective(ref_img, translation, mosaic_size)
    mask2 = (img2_warped != 0).any(axis=2)
    mosaic[mask2] = img2_warped[mask2]
    
    img3_warped = cv2.warpPerspective(images[2], translation @ H23_inv, mosaic_size)
    mask3 = (img3_warped != 0).any(axis=2)
    mask3_new = mask3 & ~(mosaic.any(axis=2))
    mosaic[mask3_new] = img3_warped[mask3_new]
    
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
    if len(sys.argv) != 6:
        print("Usage: python stitching_3.py <image-1-path> <image-2-path> <image-3-path> <H12-path> <H23-path>")
        sys.exit(1)
    
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    img3_path = sys.argv[3]
    H12_path = sys.argv[4]
    H23_path = sys.argv[5]
    
    images = [
        cv2.imread(img1_path),
        cv2.imread(img2_path),
        cv2.imread(img3_path)
    ]
    
    H12 = np.load(H12_path)
    H23 = np.load(H23_path)
    homographies = [H12, H23]
    
    mosaic = warp_and_composite_multiple(images, homographies, ref_idx=1)
    
    final_mosaic = remove_black_borders(mosaic)
    
    cv2.imwrite("stitched_3_images.jpg", final_mosaic)
    print("Stitched image saved as 'stitched_3_images.jpg'")
    
    cv2.imshow("Mosaic", final_mosaic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()