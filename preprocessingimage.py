import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_dialysate_region(image_path, window_size=250):
    # Load image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Unable to load image from {image_path}")
        return
    
    # Convert to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Use OpenCV's built-in ROI selector to select the dialysate area manually
    roi = cv2.selectROI("Select Dialysate Area", img)
    
    # Check if ROI is valid (width and height should be greater than 0)
    if roi[2] == 0 or roi[3] == 0:
        print("Error: Invalid ROI selected. Please select a valid region.")
        return
    
    # Extract the ROI region from the HSV image
    roi_hsv = hsv_img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    
    # Check if the selected ROI is valid (non-zero size array)
    if roi_hsv.size == 0:
        print("Error: The selected ROI is empty. Please select a valid region.")
        return
    
    # Calculate the minimum and maximum HSV values within the selected region
    min_hue = np.min(roi_hsv[:, :, 0])
    max_hue = np.max(roi_hsv[:, :, 0])

    min_saturation = np.min(roi_hsv[:, :, 1])
    max_saturation = np.max(roi_hsv[:, :, 1])

    min_value = np.min(roi_hsv[:, :, 2])
    max_value = np.max(roi_hsv[:, :, 2])

    # Print out the HSV bounds
    print(f"Lower Bound HSV: [{min_hue}, {min_saturation}, {min_value}]")
    print(f"Upper Bound HSV: [{max_hue}, {max_saturation}, {max_value}]")
    
    # Close the ROI window
    cv2.destroyAllWindows()

    # Split the HSV image into individual channels
    hue, saturation, value = cv2.split(hsv_img)

    # Display the original image and each channel using Matplotlib
    plt.figure(figsize=(10, 8))

    # Original image in BGR format
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    # Hue channel
    plt.subplot(2, 2, 2)
    plt.imshow(hue, cmap='hsv')
    plt.title("Hue Channel")
    plt.axis('off')

    # Saturation channel
    plt.subplot(2, 2, 3)
    plt.imshow(saturation, cmap='gray')
    plt.title("Saturation Channel")
    plt.axis('off')

    # Value channel (brightness)
    plt.subplot(2, 2, 4)
    plt.imshow(value, cmap='gray')
    plt.title("Value Channel")
    plt.axis('off')

    # Show the plots
    plt.tight_layout()
    plt.show()

    # Step 1: Create a Mask to Isolate the Dialysate Region
    lower_bound = np.array([min_hue, min_saturation, min_value])  # Calculated lower bound for dialysate color
    upper_bound = np.array([max_hue, max_saturation, max_value])  # Calculated upper bound for dialysate color

    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Step 2: Use the Mask to Restrict Segmentation to the Dialysate Region
    h, w = img.shape[:2]
    clean_squares = []
    
    for y in range(0, h - window_size, window_size):
        for x in range(0, w - window_size, window_size):
            # Check if the square is within the masked region (dialysate)
            square_mask = mask[y:y + window_size, x:x + window_size]
            
            if np.all(square_mask > 0):  # Ensure the entire square is within the mask
                square = img[y:y + window_size, x:x + window_size]
                
                # Perform further checks for glossiness and bubbles here (if needed)
                # If valid, add the clean square to the list
                clean_squares.append(square)
                # Optionally, mark the valid regions on the original image
                cv2.rectangle(img, (x, y), (x + window_size, y + window_size), (0, 255, 0), 2)
    
    # Step 3: Display and Save the Results
    cv2.imshow('Detected Dialysate Region', img)
    cv2.waitKey(0)
    
    # Save clean squares (you can adjust this if needed)
    for idx, square in enumerate(clean_squares):
        cv2.imshow(f'clean_glossy_region_{idx}.png', square)
    
    return clean_squares


# Example usage
clean_squares = detect_dialysate_region('/Users/zhangjinxun/Documents/Research/experiment/PreliminaryTraining/lib/SimulateDialysate/train_set/uninfected/IMG_5818.JPG')
