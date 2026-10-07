# SVD-Based Grayscale Image Compression

## Overview

This project demonstrates grayscale image compression using Singular Value Decomposition (SVD) in Python.

A grayscale image can be represented as a matrix where each element represents the intensity of a pixel from 0 (black) to 255 (white).

SVD decomposes an image matrix A into three matrices:

A = UΣVᵀ

By keeping only the most important singular values and their corresponding singular vectors, we can reconstruct an approximation of the original image using fewer components.

This allows us to study the trade-off between image quality and compression.

---

## P2 — Grayscale Compression and Visual Comparison

This repository contains the P2 component of the project.

The main focus is applying SVD to a grayscale image and visually comparing reconstructions obtained using different values of k.

### What the program does

1. Loads an input image.
2. Converts the image to grayscale.
3. Resizes it to 512 × 512 pixels.
4. Converts the image into a NumPy matrix.
5. Performs Singular Value Decomposition.
6. Reconstructs the image using different values of k.
7. Displays the original and reconstructed images side-by-side.
8. Plots the singular-value decay.

---

## SVD and Rank-k Approximation

For an image matrix A, SVD gives:

A = UΣVᵀ

The singular values in Σ are arranged from largest to smallest.

Instead of using all singular values, we can keep only the first k:

Aₖ = UₖΣₖVₖᵀ

This produces a rank-k approximation of the original image.

The value of k controls the trade-off between compression and image quality.

| k | Expected Result |
|---|---|
| 5 | Very blurry |
| 20 | Recognizable |
| 50 | Good quality |
| 100 | Much sharper |

A smaller k uses fewer components and therefore provides greater compression, while a larger k retains more information and produces a reconstruction closer to the original.

---

## Why SVD Compression Works

Natural images contain a significant amount of redundancy.

The largest singular values generally capture the most important structures and intensity patterns in the image, while smaller singular values often represent finer details.

Therefore, many images can be approximated reasonably well using only a subset of their singular components.

The singular-value decay plot helps visualize this behavior.

---

## Singular Value Decay

The program plots the singular values in descending order.

A rapid decrease indicates that a relatively small number of singular components contain a large amount of the image's information.

This helps explain why low-rank approximations can be useful for image compression.

---

## Technologies Used

- Python
- NumPy
- Matplotlib
- Pillow
- Singular Value Decomposition

---

## Installation

Install the required Python libraries:

```bash
pip install numpy matplotlib pillow
