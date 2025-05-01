👤 Eigenfaces & ⭕ Ellipse Detection – Image Processing Project
📌 Overview
This project consists of two core image processing tasks:

Eigenfaces for Facial Recognition – Implemented using Principal Component Analysis (PCA) to reduce dimensionality and reconstruct face images from the Olivetti dataset.

Ellipse Detection – A custom method to detect elliptical shapes in images using edge detection and Hough Transform-style voting.

Both components demonstrate key techniques in pattern recognition and computer vision, applied to practical use cases like biometric identification and shape analysis.

🧠 Project Objectives
Eigenfaces
Perform facial recognition using PCA.

Visualize top-K Eigenfaces and image reconstruction at varying principal component levels.

Analyze variance and the efficiency of data compression.

Ellipse Detection
Detect elliptical features in images (e.g., eyes, wheels, cells).

Tune detection parameters for improved performance and visualization.

Optimize accuracy, reduce false positives, and improve visual clarity.

🛠️ Technologies Used
Python

Scikit-learn (Olivetti Faces Dataset)

NumPy

OpenCV

Matplotlib

⚙️ Parameters
Eigenfaces
TopK: Number of top eigenfaces to display.

ImageNum: Index of the image to reconstruct.

Ellipse Detection
Image: Input image.

k: Kernel size for Gaussian blur.

sigma: Standard deviation for blur.

a_low, a_high: Range for semi-major axis.

b_low, b_high: Range for semi-minor axis.

🖼️ Visual Outputs
Eigenfaces
Mean face

Top K Eigenfaces

Original vs. Reconstructed images (at 10%, 15%, 30%, 60% variance)

Ellipse Detection
Ellipses drawn over detected features

Red-colored overlay for visualization

Adjustable scaling and accuracy thresholds
