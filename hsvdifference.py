# the HSV (Hue, Saturation, Value) color space is often more aligned with 
# how humans perceive and describe color. 

import cv2
import numpy as np

# Function to convert images to HSV and show each channel separately
def show_images_with_hsv_channels(image1, image2):
    # Convert both images to HSV color space
    hsv_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    hsv_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

    # Split the HSV images into their channels (H, S, V)
    h1, s1, v1 = cv2.split(hsv_image1)
    h2, s2, v2 = cv2.split(hsv_image2)

    # Stack the original images side by side
    stack_originals = np.hstack((image1, image2))

    # Stack the Hue channels side by side
    stack_hue = np.hstack((h1, h2))

    # Stack the Saturation channels side by side
    stack_saturation = np.hstack((s1, s2))

    # Stack the Value channels side by side
    stack_value = np.hstack((v1, v2))

    # Stack everything vertically: Original, Hue, Saturation, Value
    combined_image = np.vstack((stack_originals, cv2.cvtColor(stack_hue, cv2.COLOR_GRAY2BGR),
                                cv2.cvtColor(stack_saturation, cv2.COLOR_GRAY2BGR),
                                cv2.cvtColor(stack_value, cv2.COLOR_GRAY2BGR)))

    # Show the result
    cv2.imshow('Original Images and HSV Channels (H, S, V)', combined_image)

    # Wait indefinitely until a key is pressed, then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load the two images
image1 = cv2.imread('cropped_fixed_IMG_5934u.JPG')  # Replace with your first image
image2 = cv2.imread('cropped_fixed_IMG_5977.JPG')  # Replace with your second image

# Check if images are loaded correctly
if image1 is None or image2 is None:
    print("Error: One or both images could not be loaded. Please check the file paths.")
else:
    # Show the images and their HSV channels
    show_images_with_hsv_channels(image1, image2)
    
    
