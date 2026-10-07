import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def rank_k(U, S, Vt, k):
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]


# Load image and convert to grayscale
img = Image.open("images/photo.jpg").convert("L").resize((512, 512))

# Convert image to a NumPy matrix
A = np.array(img, dtype=float)

print("Matrix shape:", A.shape)

# Perform Singular Value Decomposition
U, S, Vt = np.linalg.svd(A, full_matrices=False)

print("Number of singular values:", len(S))

# Values of k for comparison
ks = [5, 20, 50, 100]

# Display original and reconstructed images
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(A, cmap="gray")
plt.title("Original")
plt.axis("off")

for i, k in enumerate(ks):
    Ak = rank_k(U, S, Vt, k)

    plt.subplot(2, 3, i + 2)
    plt.imshow(np.clip(Ak, 0, 255), cmap="gray")
    plt.title(f"k = {k}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# Plot singular value decay
plt.figure(figsize=(8, 5))
plt.plot(S)
plt.title("Singular Value Decay")
plt.xlabel("Singular Value Index")
plt.ylabel("Singular Value")
plt.yscale("log")
plt.grid()

plt.show()
