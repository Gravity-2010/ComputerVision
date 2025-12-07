import cv2
import numpy as np
import matplotlib.pyplot as plt
from homography import detect_and_describe_sift, match_features, ransac

def visualize_keypoints(img, keypoints, title="Detected Keypoints"):
    img_with_kp = cv2.drawKeypoints(img, keypoints, None, 
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
    plt.title(f"{title} ({len(keypoints)} keypoints)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=150, bbox_inches='tight')
    plt.show()


def visualize_matches(img1, img2, kp1, kp2, matches, title="Feature Matches"):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f"{title} ({len(matches)} matches)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=150, bbox_inches='tight')
    plt.show()


def visualize_inliers(img1, img2, kp1, kp2, matches, inlier_indices, 
                      title_tentative="Tentative Matches",
                      title_inliers="RANSAC Inliers"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    img_tentative = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    ax1.imshow(cv2.cvtColor(img_tentative, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"{title_tentative} ({len(matches)} matches)")
    ax1.axis('off')
    
    inlier_matches = [matches[i] for i in inlier_indices]
    img_inliers = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None,
                                   matchColor=(0, 255, 0),
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    ax2.imshow(cv2.cvtColor(img_inliers, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"{title_inliers} ({len(inlier_matches)} inliers)")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig("tentative_and_inliers.png", dpi=150, bbox_inches='tight')
    plt.show()


def create_report_visualizations(img1_path, img2_path, output_prefix="keble"):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    kp1, desc1 = detect_and_describe_sift(img1)
    kp2, desc2 = detect_and_describe_sift(img2)
    
    print(f"Image 1: {len(kp1)} keypoints")
    print(f"Image 2: {len(kp2)} keypoints")
    
    visualize_keypoints(img1, kp1, f"{output_prefix}_img1_keypoints")
    visualize_keypoints(img2, kp2, f"{output_prefix}_img2_keypoints")
    
    matches = match_features(desc1, desc2, ratio_thresh=0.75)
    print(f"Tentative matches: {len(matches)}")
    
    visualize_matches(img1, img2, kp1, kp2, matches, f"{output_prefix}_tentative_matches")
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).T
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).T
    corr = {'pts1': pts1, 'pts2': pts2}
    H, inliers = ransac(corr, thresh=5.0, num_iter=2000)
    print(f"RANSAC inliers: {len(inliers)}")
    
    visualize_inliers(img1, img2, kp1, kp2, matches, inliers,
                     f"{output_prefix}_tentative", f"{output_prefix}_inliers")
 
    return H, len(matches), len(inliers)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python visualize_matches.py <image-1-path> <image-2-path>")
        sys.exit(1)
    
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    
    H, n_tentative, n_inliers = create_report_visualizations(img1_path, img2_path)
    
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"Tentative matches: {n_tentative}")
    print(f"RANSAC inliers: {n_inliers}")
    print(f"Inlier ratio: {n_inliers/n_tentative*100:.1f}%")
    print(f"Estimated Homography:\n{H}")
    print(f"{'='*50}")