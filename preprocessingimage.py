import cv2
import numpy as np

# Load the original image
image = cv2.imread('/Users/zhangjinxun/Documents/Research/experiment/PreliminaryTraining/lib/SimulateDialysate/predict/IMG_5834u.JPG')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Adaptive thresholding to detect the main sample area (the dialysate bag)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Use morphological operations to clean up noise in the thresholded image
kernel = np.ones((5, 5), np.uint8)
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Find contours to detect the main dialysate bag
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)

# Create a mask for the detected sample area
mask = np.zeros_like(gray)
cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

# Mask the original image to isolate the sample area and avoid the background
sample_region = cv2.bitwise_and(image, image, mask=mask)

# --- Bubble Detection ---
sample_gray = cv2.cvtColor(sample_region, cv2.COLOR_BGR2GRAY)
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 20  # Lowered area threshold to detect smaller bubbles
params.filterByCircularity = True
params.minCircularity = 0.7  # Adjust circularity to detect less perfect bubbles
params.filterByInertia = True
params.minInertiaRatio = 0.3
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(sample_gray)

# Detect reflections using Canny edge detection and thresholding
gray_sample = cv2.cvtColor(sample_region, cv2.COLOR_BGR2GRAY)
_, reflections_mask = cv2.threshold(gray_sample, 180, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(gray_sample, 50, 150)
reflection_mask = cv2.bitwise_or(reflections_mask, edges)

# --- Combine Bubble and Reflection Masks ---
bubble_mask = np.zeros_like(gray_sample)
for kp in keypoints:
    cv2.circle(bubble_mask, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size/2), 255, thickness=cv2.FILLED)

combined_mask = cv2.bitwise_or(bubble_mask, reflection_mask)

# --- Fixed-Size Square Segmentation Inside the Masked Region ---
patch_size = 100  # Define the size of the square patch
height, width = sample_region.shape[:2]

patch_count = 1
for y in range(0, height, patch_size):
    for x in range(0, width, patch_size):
        # Ensure we don't exceed image bounds
        if y + patch_size <= height and x + patch_size <= width:
            patch = sample_region[y:y + patch_size, x:x + patch_size]

            # Check if the patch contains bubbles or reflections by using the combined mask
            patch_mask = combined_mask[y:y + patch_size, x:x + patch_size]
            
            # Only consider patches that are fully inside the sample mask (avoid background)
            sample_mask_patch = mask[y:y + patch_size, x:x + patch_size]
            if np.sum(patch_mask) == 0 and np.sum(sample_mask_patch) > 0:  # If no bubbles/reflections & inside mask
                cv2.imshow(f'Clean Patch {patch_count}', patch)
                patch_count += 1

# Wait for key press to close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
