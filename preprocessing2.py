import cv2

# Global variables
ref_point = []
cropping = False
fixed_size = (300, 300)  # Fixed size for cutting (width, height)

# Function to crop the image to the fixed size
def crop_fixed_size(image, start_point, size):
    x1, y1 = start_point
    width, height = size

    # Ensure the cropped region doesn't exceed the image boundaries
    x2 = min(x1 + width, image.shape[1])
    y2 = min(y1 + height, image.shape[0])

    # Crop the region with fixed size
    cropped_image = image[y1:y2, x1:x2]

    return cropped_image

# Mouse callback function for selecting the region
def mouse_crop(event, x, y, flags, param):
    global ref_point, cropping

    # On left button click, record the top-left point
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    # On left button release, crop the fixed size region
    elif event == cv2.EVENT_LBUTTONUP:
        cropping = False
        ref_point.append((x, y))

        # Crop the region with fixed size starting from the selected point
        cropped_image = crop_fixed_size(image, ref_point[0], fixed_size)

        # Draw the rectangle on the original image (for visual feedback)
        x1, y1 = ref_point[0]
        x2, y2 = x1 + fixed_size[0], y1 + fixed_size[1]
        cv2.rectangle(image, (x1, y1), (min(x2, image.shape[1]), min(y2, image.shape[0])), (0, 255, 0), 2)
        cv2.imshow("image", image)

        # Show the cropped region in a new window
        cv2.imshow("Cropped Image", cropped_image)
        cv2.imwrite("cropped_fixed_image.jpg", cropped_image)  # Save result to file

# Load the image
image = cv2.imread('image.jpg')

# Create a window and set the mouse callback function
cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_crop)

# Show the image and wait for the user to select the region
while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # If 'r' key is pressed, reset the cropping
    if key == ord("r"):
        image = cv2.imread('image.jpg')  # Reload the original image

    # If 'q' key is pressed, exit the program
    elif key == ord("q"):
        break

# Destroy all windows when done
cv2.destroyAllWindows()
