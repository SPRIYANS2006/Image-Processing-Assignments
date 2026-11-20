# Assignment 5 — Histogram Equalization and Matching

## Question 1: Histogram Equalization

Write a program to enhance the contrast of the input grayscale image using histogram equalization. Display the original and equalized images.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
hist = np.zeros(256, dtype=int)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        hist[img[i, j]] += 1

cdf = np.zeros(256, dtype=int)
cdf[0] = hist[0]
for i in range(1, 256):
    cdf[i] = cdf[i-1] + hist[i]

cdf_min = min(cdf[cdf > 0])
total = img.shape[0] * img.shape[1]
mapping = np.zeros(256, dtype=np.uint8)
for i in range(256):
    mapping[i] = int(((cdf[i] - cdf_min) / (total - cdf_min)) * 255)

result = np.zeros(img.shape, dtype=np.uint8)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        result[i, j] = mapping[img[i, j]]

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("Histogram Equalized Image")
plt.axis("off")
plt.show()
```

## Question 2: Histogram Matching

Write a program to perform histogram matching of the input image with a reference image using histogram equalization.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
ref = cv2.imread("reference.jpg", cv2.IMREAD_GRAYSCALE)
if img is None or ref is None:
    print("Image not found!")
    exit()

hist1 = np.zeros(256, dtype=int)
hist2 = np.zeros(256, dtype=int)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        hist1[img[i, j]] += 1
for i in range(ref.shape[0]):
    for j in range(ref.shape[1]):
        hist2[ref[i, j]] += 1

cdf1 = np.cumsum(hist1) / img.size
cdf2 = np.cumsum(hist2) / ref.size
mapping = np.zeros(256, dtype=np.uint8)
for i in range(256):
    diff = np.abs(cdf2 - cdf1[i])
    mapping[i] = np.argmin(diff)

result = np.zeros(img.shape, dtype=np.uint8)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        result[i, j] = mapping[img[i, j]]

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Input Image")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(ref, cmap="gray")
plt.title("Reference Image")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(result, cmap="gray")
plt.title("Histogram Matched Image")
plt.axis("off")
plt.show()
```
