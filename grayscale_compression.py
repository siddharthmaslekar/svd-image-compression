import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def rank_k(U, S, Vt, k):
    """Reconstruct the image using the first k singular components."""
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]


# --------------------------------------------------
# 1. Load and prepare image
# --------------------------------------------------

img = Image.open("images/demo.jpg").convert("L").resize((512, 512))
A = np.array(img, dtype=float)

m, n = A.shape

print("======================================")
print("   SVD GRAYSCALE IMAGE COMPRESSION")
print("======================================")
print(f"Image size: {m} x {n}")
print(f"Original matrix storage: {m * n} values")


# --------------------------------------------------
# 2. Perform SVD
# --------------------------------------------------

U, S, Vt = np.linalg.svd(A, full_matrices=False)

print(f"Number of singular values: {len(S)}")


# --------------------------------------------------
# 3. Create output directory
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)


# --------------------------------------------------
# 4. Rank-k reconstruction comparison
# --------------------------------------------------

ks = [5, 20, 50, 100]

plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
plt.imshow(A, cmap="gray")
plt.title("Original")
plt.axis("off")


print("\nStorage comparison:")
print("-" * 65)
print(f"{'k':<10}{'Values Stored':<20}{'Compression Ratio':<20}{'Storage Saved'}")
print("-" * 65)

for i, k in enumerate(ks):

    Ak = rank_k(U, S, Vt, k)

    plt.subplot(2, 3, i + 2)
    plt.imshow(np.clip(Ak, 0, 255), cmap="gray")
    plt.title(f"k = {k}")
    plt.axis("off")

    # Storage required by rank-k representation
    compressed_storage = k * (m + n + 1)

    # Compression ratio
    compression_ratio = (m * n) / compressed_storage

    # Percentage storage saved
    storage_saved = (1 - compressed_storage / (m * n)) * 100

    print(
        f"{k:<10}"
        f"{compressed_storage:<20}"
        f"{compression_ratio:.2f}x{'':<17}"
        f"{storage_saved:.2f}%"
    )


plt.suptitle("SVD Rank-k Grayscale Image Compression", fontsize=16)
plt.tight_layout()

# Save comparison figure
plt.savefig(
    "outputs/grayscale_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# --------------------------------------------------
# 5. Singular-value decay plot
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(S)

plt.title("Singular Value Decay")
plt.xlabel("Singular Value Index")
plt.ylabel("Singular Value")

# Log scale makes the large range easier to see
plt.yscale("log")

plt.grid()

# Save singular-value plot
plt.savefig(
    "outputs/singular_value_decay.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\n======================================")
print("Output files saved in the 'outputs' folder:")
print("  - grayscale_comparison.png")
print("  - singular_value_decay.png")
print("======================================")
