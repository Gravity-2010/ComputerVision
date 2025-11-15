import cv2
import numpy as np
import random
import os
import matplotlib.pyplot as plt

folder = 'keble_images'
image_files = ['keble_a.jpg', 'keble_b.jpg', 'keble_c.jpg']
images = [cv2.imread(os.path.join(folder, f)) for f in image_files]

ref_idx = len(images) // 2
ref_img = images[ref_idx]
ref_h, ref_w = ref_img.shape[:2]

# Detect SIFT features
sift = cv2.SIFT_create()
ref_kp, ref_des = sift.detectAndCompute(ref_img, None)

# Feature matching
def match_features(des1, des2, ratio=0.75):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches

# Visualization functions with image title
def draw_matches(img1, kp1, img2, kp2, matches, img_name):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                  matchColor=(0,255,0), singlePointColor=None, flags=2)
    plt.figure(figsize=(12,8))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Tentative Correspondences: {img_name} and Reference")
    plt.show()

def draw_displacements(img1, kp1, img2, kp2, matches, img_name):
    img_disp = img1.copy()
    for m in matches:
        pt1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
        pt2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int))
        cv2.circle(img_disp, pt1, 4, (0,255,0), -1)  
        cv2.line(img_disp, pt1, pt2, (255,0,0), 1) 
    plt.figure(figsize=(10,8))
    plt.imshow(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Feature Displacements: {img_name} and Reference")
    plt.show()

def draw_inliers(img1, img2, kp1, kp2, matches, inlier_idx, img_name):
    inlier_matches = [matches[i] for i in inlier_idx]
    img_inliers = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None,
                                  matchColor=(0,255,0), singlePointColor=None, flags=2)
    plt.figure(figsize=(12,8))
    plt.imshow(cv2.cvtColor(img_inliers, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Inlier Matches After RANSAC: {img_name} and Reference")
    plt.show()

# Compute homography
def computeH(pts1, pts2):
    n = pts1.shape[0]
    A = []
    for i in range(n):
        x, y = pts1[i][0], pts1[i][1]
        xp, yp = pts2[i][0], pts2[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3,3)
    return H / H[2,2]

# RANSAC
def ransac(matches, kp1, kp2, thresh=5.0, iterations=1000):
    max_inliers = []
    final_H = None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    for _ in range(iterations):
        idx = random.sample(range(len(matches)), 4)
        H = computeH(pts1[idx], pts2[idx])
        pts1_hom = np.hstack((pts1, np.ones((pts1.shape[0],1))))
        pts2_proj = (H @ pts1_hom.T).T
        pts2_proj /= pts2_proj[:,2].reshape(-1,1)
        errors = np.linalg.norm(pts2 - pts2_proj[:,0:2], axis=1)
        inliers = np.where(errors < thresh)[0]
        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            final_H = H
    if len(max_inliers) >= 4:
        final_H = computeH(pts1[max_inliers], pts2[max_inliers])
    return final_H, max_inliers

# Stitching
mosaic = ref_img.copy()
for idx, img in enumerate(images):
    if idx == ref_idx:
        continue
    kp, des = sift.detectAndCompute(img, None)
    matches = match_features(des, ref_des)

    # Use image filename as title
    img_name = image_files[idx]

    draw_matches(img, kp, ref_img, ref_kp, matches, img_name)
    draw_displacements(img, kp, ref_img, ref_kp, matches, img_name)

    H, inliers = ransac(matches, kp, ref_kp)

    draw_inliers(img, ref_img, kp, ref_kp, matches, inliers, img_name)

    h, w = ref_img.shape[:2]
    warped = cv2.warpPerspective(img, H, (w*2, h*2))
    mosaic = cv2.addWeighted(mosaic, 1, warped[0:h,0:w], 1, 0)

# Final mosaic display
mosaic_rgb = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12,8))
plt.imshow(mosaic_rgb)
plt.axis('off')
plt.title("Stitched Mosaic")
plt.show()
