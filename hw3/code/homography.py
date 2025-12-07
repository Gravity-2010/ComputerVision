import cv2
import numpy as np
import sys

def detect_and_describe_sift(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(desc1, desc2, ratio_thresh=0.75):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
    
    return good_matches


def computeH(im1_pts, im2_pts):
    n = im1_pts.shape[1]
    
    A = []
    for i in range(n):
        x, y = im1_pts[0, i], im1_pts[1, i]
        x_prime, y_prime = im2_pts[0, i], im2_pts[1, i]
        
        A.append([-x, -y, -1, 0, 0, 0, x*x_prime, y*x_prime, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x*y_prime, y*y_prime, y_prime])
    
    A = np.array(A)
    
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :] 
    
    H = h.reshape(3, 3)
    
    return H


def ransac(corr, thresh=5.0, num_iter=1000):
    pts1 = corr['pts1']
    pts2 = corr['pts2']
    n = pts1.shape[1]
    
    best_H = None
    max_inliers = []
    
    for _ in range(num_iter):
        indices = np.random.choice(n, 4, replace=False)
        sample_pts1 = pts1[:, indices]
        sample_pts2 = pts2[:, indices]
        
        H = computeH(sample_pts1, sample_pts2)
        
        pts1_homog = np.vstack([pts1, np.ones((1, n))])
        pts2_pred_homog = H @ pts1_homog
        
        pts2_pred = pts2_pred_homog[:2, :] / pts2_pred_homog[2, :]
        
        distances = np.sqrt(np.sum((pts2 - pts2_pred)**2, axis=0))
        
        inliers = np.where(distances < thresh)[0]
        
        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            best_H = H
    
    if len(max_inliers) >= 4:
        inlier_pts1 = pts1[:, max_inliers]
        inlier_pts2 = pts2[:, max_inliers]
        best_H = computeH(inlier_pts1, inlier_pts2)
    
    return best_H, max_inliers


def estimate_homography(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    kp1, desc1 = detect_and_describe_sift(img1)
    kp2, desc2 = detect_and_describe_sift(img2)
    
    matches = match_features(desc1, desc2, ratio_thresh=0.75)
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).T  
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).T  
    
    corr = {'pts1': pts1, 'pts2': pts2}
    H, inliers = ransac(corr, thresh=5.0, num_iter=2000)
    
    return H, kp1, kp2, matches, inliers


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python homography.py <image-1-path> <image-2-path> <H-path>")
        sys.exit(1)
    
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    H_path = sys.argv[3]
    
    H, _, _, _, _ = estimate_homography(img1_path, img2_path)
    
    np.save(H_path, H)
    print(f"Homography matrix saved to {H_path}")
    print(f"H shape: {H.shape}")
    print(f"H:\n{H}")