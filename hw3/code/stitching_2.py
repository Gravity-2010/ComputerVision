import cv2
import numpy as np
import sys

def warp_and_composite(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    
    corners1_transformed = cv2.perspectiveTransform(corners1, H)
    
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    
    all_corners = np.concatenate((corners1_transformed, corners2), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    translation = np.array([[1, 0, -x_min],
                           [0, 1, -y_min],
                           [0, 0, 1]], dtype=np.float32)
    
    mosaic_size = (x_max - x_min, y_max - y_min)
    img1_warped = cv2.warpPerspective(img1, translation @ H, mosaic_size)
    
    mosaic = np.zeros((mosaic_size[1], mosaic_size[0], 3), dtype=np.uint8)
    mosaic[-y_min:-y_min+h2, -x_min:-x_min+w2] = img2
    
    mask = (mosaic == 0).all(axis=2)
    mosaic[mask] = img1_warped[mask]
    
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
    if len(sys.argv) != 4:
        print("Usage: python stitching_2.py <image-1-path> <image-2-path> <H12-path>")
        sys.exit(1)
    
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    H12_path = sys.argv[3]
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    H12 = np.load(H12_path)
    
    mosaic = warp_and_composite(img1, img2, H12)
    
    final_mosaic = remove_black_borders(mosaic)
    
    cv2.imwrite("stitched_2_images.jpg", final_mosaic)
    print("Stitched image saved as 'stitched_2_images.jpg'")
    
    cv2.imshow("Mosaic", final_mosaic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()