
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces


def eigenFaces(topK, imageNum):
    # Read the Olivetti Faces dataset
    faces = fetch_olivetti_faces()
    images = faces.images

    # Flatten the images to create a matrix where each row represents a flattened image
    X = np.array([img.flatten() for img in images])

    # Calculate the mean face
    mean_face = np.mean(X, axis=0)

    # Subtract the mean face from each image
    X_centered = X - mean_face

    # Calculate the covariance matrix
    covariance_matrix = np.cov(X_centered, rowvar=False)

    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Choose the top k eigenvectors (eigenfaces)
    num_eigenfaces = topK
    eigenfaces = eigenvectors[:, :num_eigenfaces]

    # Calculate the cumulative variance explained by the eigenvalues
    cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)

    # Define variance percentages for reconstruction
    variance_percentages = [0.10, 0.15, 0.30, 0.60]

    # Find the number of components for each variance percentage
    num_components = [np.argmax(cumulative_variance >= vp) +
                      1 for vp in variance_percentages]

    # Reconstruct a sample image using the selected components
    sample_image_index = imageNum  # Use the first image as an example
    reconstructions = []

    for n in num_components:
        eigenfaces_n = eigenvectors[:, :n]
        weights = np.dot(X_centered[sample_image_index], eigenfaces_n)
        reconstruction = np.dot(weights, eigenfaces_n.T) + mean_face
        reconstructions.append(reconstruction)

    # Create a combined figure to display eigenfaces and reconstructed images
    fig, axes = plt.subplots(7, 8, figsize=(15, 20))

    # Display the top k eigenfaces
    for i, ax in enumerate(axes[:5].flat):
        ax.imshow(eigenfaces[:, i].reshape(64, 64), cmap='gray')
        ax.axis('off')

    # Display the original and reconstructed images
    axes[5, 0].imshow(images[sample_image_index], cmap='gray')
    axes[5, 0].axis('off')
    axes[5, 0].text(0.5, -0.15, 'Original', ha='center',
                    va='top', transform=axes[5, 0].transAxes)

    for i, (reconstruction, n) in enumerate(zip(reconstructions, num_components)):
        ax = axes[5 + (i // 4), (i % 4) + 1]
        ax.imshow(reconstruction.reshape(64, 64), cmap='gray')
        ax.axis('off')
        ax.text(0.5, -0.15, f'{n} components', ha='center',
                va='top', transform=ax.transAxes)

    # Remove any remaining unused subplots
    for j in range(len(reconstructions) + 1, 8):
        axes[5, j].axis('off')
    for j in range(8):
        axes[6, j].axis('off')

    # Adjust layout
    # Increase the height and width spacing between subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.tight_layout()
    plt.show()
