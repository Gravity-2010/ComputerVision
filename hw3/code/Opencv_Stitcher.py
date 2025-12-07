import cv2
import sys

def stitch_with_opencv(image_paths, output_path="opencv_panorama.jpg"):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Error: Could not load {path}")
            return None, None
        images.append(img)
    
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    
    status, panorama = stitcher.stitch(images)
    
    if status == cv2.Stitcher_OK:
        cv2.imwrite(output_path, panorama)
        print(f"Panorama saved to {output_path}")
        return status, panorama
    else:
        print(f"Stitching failed with status: {status}")
        return status, None


if __name__ == "__main__":
    image_paths = [
        '../images/keble_images/keble_a.jpg',
        '../images/keble_images/keble_b.jpg',
        '../images/keble_images/keble_c.jpg'
    ]
    
    status, panorama = stitch_with_opencv(image_paths, 'opencv_keble_panorama.jpg')