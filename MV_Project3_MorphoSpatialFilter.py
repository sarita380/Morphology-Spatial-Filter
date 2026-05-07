import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt



# -----------------------------
# Helper: safe image loading
# -----------------------------
BASE_DIR = r"C:\Users\sagri\PycharmProjects\Project3_Morpho_SpatialFilter"

moon_path = os.path.join(BASE_DIR, "moon.jpg")
fingerprint_path = os.path.join(BASE_DIR, "fingerprint.jpg")
cell_path = os.path.join(BASE_DIR, "cell.jpg")

moon = cv2.imread(moon_path, cv2.IMREAD_GRAYSCALE)
fingerprint = cv2.imread(fingerprint_path, cv2.IMREAD_GRAYSCALE)
cell = cv2.imread(cell_path)

if moon is None or fingerprint is None or cell is None:
    print("Error loading one or more images.")
    exit()

# -----------------------------
# 1) Moon: Laplace and Sobel
# -----------------------------
lap = cv2.Laplacian(moon, cv2.CV_64F, ksize=3)
lap_abs = cv2.convertScaleAbs(lap)
moon_lap_sharp = cv2.addWeighted(moon, 1.0, lap_abs, 0.7, 0)

sobelx = cv2.Sobel(moon, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(moon, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = cv2.magnitude(sobelx, sobely)
sobel_abs = cv2.convertScaleAbs(sobel_mag)
moon_sobel_sharp = cv2.addWeighted(moon, 1.0, sobel_abs, 0.7, 0)

# -----------------------------
# 2) Fingerprint: threshold + morphology + histogram
# -----------------------------
threshold_value = 160
_, fp_binary = cv2.threshold(fingerprint, threshold_value, 255, cv2.THRESH_BINARY_INV)

kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

fp_open = cv2.morphologyEx(fp_binary, cv2.MORPH_OPEN, kernel_open)
fp_clean = cv2.morphologyEx(fp_open, cv2.MORPH_CLOSE, kernel_close)

# -----------------------------
# 3) Cells: count, area, biggest boundary
# -----------------------------
cell_rgb = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)

hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
lower_green = np.array([35, 40, 40])
upper_green = np.array([90, 255, 255])
green_mask = cv2.inRange(hsv, lower_green, upper_green)

kernel = np.ones((3, 3), np.uint8)
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(green_mask, connectivity=8)

areas = []
largest_idx = -1
largest_area = 0
cell_count = 0

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if area > 500:
        areas.append(int(area))
        cell_count += 1
        if area > largest_area:
            largest_area = area
            largest_idx = i

output_cells = cell_rgb.copy()

if largest_idx != -1:
    largest_mask = np.uint8(labels == largest_idx) * 255
    contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output_cells, contours, -1, (255, 255, 0), 3)

print("Detected cells:", cell_count)
print("Cell areas (pixels):", areas)
print("Fingerprint threshold:", threshold_value)

# -----------------------------
# DISPLAY RESULTS
# -----------------------------
plt.figure(figsize=(16, 12))

# Moon comparison
plt.subplot(4, 3, 1)
plt.imshow(moon, cmap='gray')
plt.title("Original Moon")
plt.axis("off")

plt.subplot(4, 3, 2)
plt.imshow(moon_lap_sharp, cmap='gray')
plt.title("Laplacian Sharpened")
plt.axis("off")

plt.subplot(4, 3, 3)
plt.imshow(moon_sobel_sharp, cmap='gray')
plt.title("Sobel Sharpened")
plt.axis("off")

# Fingerprint comparison
plt.subplot(4, 3, 4)
plt.imshow(fingerprint, cmap='gray')
plt.title("Original Fingerprint")
plt.axis("off")

plt.subplot(4, 3, 5)
plt.imshow(fp_binary, cmap='gray')
plt.title("Binary Fingerprint")
plt.axis("off")

plt.subplot(4, 3, 6)
plt.imshow(fp_clean, cmap='gray')
plt.title("Morphology Cleaned")
plt.axis("off")

# Histogram for part 2
plt.subplot(4, 3, 7)
plt.hist(fingerprint.ravel(), bins=256, range=[0, 256], color='gray')
plt.axvline(threshold_value, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold_value}')
plt.title("Fingerprint Histogram")
plt.xlabel("Gray level")
plt.ylabel("Pixel count")
plt.legend()

# Optional morphology step view
plt.subplot(4, 3, 8)
plt.imshow(fp_open, cmap='gray')
plt.title("After Opening")
plt.axis("off")

plt.subplot(4, 3, 9)
plt.imshow(fp_clean, cmap='gray')
plt.title("After Closing")
plt.axis("off")

# Cell comparison
plt.subplot(4, 3, 10)
plt.imshow(cell_rgb)
plt.title("Original Cells")
plt.axis("off")

plt.subplot(4, 3, 11)
plt.imshow(green_mask, cmap='gray')
plt.title("Cell Mask")
plt.axis("off")

plt.subplot(4, 3, 12)
plt.imshow(output_cells)
plt.title("Biggest Cell Boundary")
plt.axis("off")

plt.tight_layout()
plt.show()
