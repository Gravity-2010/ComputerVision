# 👁️ Computer Vision Coursework

### Classical Image Processing, Feature Geometry & Image Stitching

> **Coursework Repository — M.S. Computer Science, UMBC**

This repository contains hands-on computer vision assignments completed as part of my graduate coursework at the **University of Maryland, Baltimore County (UMBC)**.

The exercises progress from foundational image-processing operations to multiresolution image blending and geometric image stitching.

The repository is preserved as coursework rather than presented as a standalone production project.

---

## 📚 Topics Covered

Across the assignments, I implemented and explored:

* Image loading and color-space conversion
* Cropping and resizing
* Downsampling and upsampling
* Nearest-neighbor and bicubic interpolation
* 2-D convolution
* Sobel edge detection
* Laplacian filtering
* Box and Gaussian blurring
* Image blending
* Hybrid images
* Gaussian pyramids
* Laplacian pyramids
* Multiresolution blending
* Corner / feature detection
* Feature matching
* Homography estimation
* Image warping
* Panorama stitching

---

# 📂 Repository Structure

```text
ComputerVision/
│
├── hw1/
│   ├── ImageBlending_Task1.py
│   ├── HybridImg_Task2.py
│   ├── hw1_Part3.py
│   └── image assets / generated outputs
│
├── hw2/
│   └── code/
│       ├── question1_task1_AlphaBlend.py
│       ├── question1_task2_GaussianLaplacianPyramids.py
│       ├── question1_task3_MultiresolutionBlending.py
│       ├── question1_task4_truegrit.py
│       ├── question2.py
│       └── image assets / outputs
│
├── hw3/
│   ├── code/
│   │   ├── homography.py
│   │   ├── stitching_2.py
│   │   ├── stitching_3.py
│   │   ├── stitching_5.py
│   │   ├── Opencv_Stitcher.py
│   │   └── visualize_matches.py
│   │
│   └── images/
│
├── .gitignore
└── README.md
```

---

# 🖼️ Assignment 1 — Image Processing & Hybrid Images

The first assignment focuses on foundational computer-vision operations.

## Image Manipulation

Implemented operations include:

* Reading and writing images with OpenCV
* BGR ↔ RGB conversion
* Grayscale conversion
* Image cropping
* Downsampling
* Upsampling

Different interpolation approaches were compared when reconstructing downsampled images, including:

```text
Nearest Neighbor
Bicubic Interpolation
```

Absolute pixel differences were also calculated to compare reconstructed images with the original.

---

## Convolution & Filtering

The assignment explores spatial image filtering using kernels.

Implemented examples include:

### Sobel Filtering

Used to highlight directional image gradients and edges.

### Laplacian Filtering

Used to emphasize rapid intensity changes.

### Box Blur

Applies an averaging kernel to smooth an image.

### Gaussian Blur

Applies Gaussian-weighted smoothing.

I also implemented a basic **2-D convolution routine manually** to better understand how spatial filtering works rather than relying exclusively on OpenCV's built-in filtering functions.

---

## Image Blending

The assignment also explores combining information from multiple images.

This includes:

* Image blending
* Frequency-aware image composition
* Hybrid-image generation

Hybrid images combine low-frequency information from one image with high-frequency information from another so perception can change depending on viewing distance.

---

# 🧩 Assignment 2 — Multiresolution Image Blending

The second assignment focuses on image representations at multiple spatial scales.

Implemented exercises include:

### Alpha Blending

Combines two images using a blending mask.

### Gaussian Pyramids

Represent an image at progressively lower spatial resolutions.

### Laplacian Pyramids

Represent image detail across multiple frequency/spatial scales.

### Multiresolution Blending

Combines images across pyramid levels to create smoother transitions than direct pixel-level blending.

The workflow can be summarized as:

```text
Input Images
     ↓
Gaussian Pyramids
     ↓
Laplacian Pyramids
     ↓
Blend Across Scales
     ↓
Reconstruct Pyramid
     ↓
Final Composite
```

The assignment also includes feature/corner-analysis exercises used to explore image structure.

---

# 🌄 Assignment 3 — Homography & Panorama Stitching

The third assignment focuses on geometric computer vision.

The implementation explores the steps required to align overlapping images and combine them into a wider scene.

```text
Input Images
     ↓
Feature Detection / Matching
     ↓
Correspondence Points
     ↓
Homography Estimation
     ↓
Perspective Warping
     ↓
Image Alignment
     ↓
Panorama Stitching
```

---

## Homography

A homography models a projective transformation between two views of approximately the same planar scene.

The transformation can be represented as:

```text
x' ~ Hx
```

where `H` is a 3 × 3 projective transformation matrix.

The assignment explores estimating and applying this transformation to align images.

---

## Image Stitching

Several stitching implementations are included to experiment with:

* correspondence visualization
* homography estimation
* image warping
* alignment
* OpenCV's stitching functionality
* combining overlapping images into panorama-style outputs

---

# 🛠️ Tech Stack

| Technology | Use                               |
| ---------- | --------------------------------- |
| Python     | Core implementation               |
| OpenCV     | Image processing and geometric CV |
| NumPy      | Matrix and numerical operations   |
| Matplotlib | Visualization                     |

---

# 🚀 Getting Started

## Clone

```bash
git clone https://github.com/Gravity-2010/ComputerVision.git
cd ComputerVision
```

## Create a virtual environment

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

## Install dependencies

```bash
pip install numpy opencv-python matplotlib
```

Individual assignment scripts may have additional requirements depending on the exercise.

---

# ▶️ Running an Exercise

For example:

```bash
cd hw1
python hw1_Part3.py
```

or:

```bash
cd hw2/code
python question1_task3_MultiresolutionBlending.py
```

Some scripts expect their associated image assets to remain in the same directory.

---

# 🎯 Learning Outcomes

These assignments provided hands-on experience with the mathematical and implementation foundations of computer vision, including:

* Understanding images as numerical arrays
* Applying convolution and spatial filters
* Comparing interpolation methods
* Reasoning about image frequency content
* Constructing multiscale image representations
* Applying projective geometry
* Estimating transformations between images
* Combining multiple views into a stitched scene

---

# ⚠️ Repository Scope

This repository contains **classical computer-vision coursework**.

It does not represent a deep-learning computer-vision project and does not currently include:

* CNN training
* object-detection models
* semantic segmentation models
* transformer-based vision models

My deep-learning computer-vision work is maintained separately from this coursework repository.

---

# 📌 Repository Status

**Completed graduate coursework**

This repository is preserved as a record of hands-on work in classical image processing and geometric computer vision.
