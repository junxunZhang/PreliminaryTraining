import cv2
import os

# Global variables
ref_point = []
cropping = False
fixed_size = (255, 255)  # Fixed size for cutting (width, height)

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

        # Show the cropped region in a new window without drawing a rectangle on the original image
        cv2.imshow("Cropped Image", cropped_image)
        cv2.imwrite(output_image_path, cropped_image)  # Save the cropped image to file
        print(f"Saved cropped image to: {output_image_path}")

# Function to loop through all images in the folder and crop them
def crop_images_in_folder(input_folder, output_folder):
    global image, output_image_path

    # Ensure output folder exists, if not create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each file in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is an image (you can extend the check for other image formats)
        if filename.endswith(".JPG") or filename.endswith(".png") or filename.endswith(".jpeg"):
            # Construct the full input file path
            input_path = os.path.join(input_folder, filename)
            
            # Load the image
            print(f"Loading image: {input_path}")
            image = cv2.imread(input_path)

            # If the image is successfully loaded, proceed to show and crop
            if image is not None:
                print(f"Loaded image successfully: {filename}")

                # Set the output image path for saving the cropped image
                output_image_path = os.path.join(output_folder, filename)

                # Create a window and set the mouse callback function for cropping
                cv2.namedWindow("image", cv2.WINDOW_NORMAL)
                cv2.setMouseCallback("image", mouse_crop)

                # Show the image and wait for the user to select the region and crop
                print(f"Showing image: {filename}")
                while True:
                    cv2.imshow("image", image)
                    key = cv2.waitKey(1) & 0xFF

                    # If 'q' key is pressed, move to the next image
                    if key == ord("q"):
                        print(f"Moving to next image")
                        break

                # Close all windows before moving to the next image
                cv2.destroyAllWindows()
            else:
                print(f"Failed to load image: {input_path}")

# Set your input and output folders
input_folder = "/Users/zhangjinxun/Documents/Research/experiment/PreliminaryTraining/lib/SimulateDialysate/predict"
output_folder = "/Users/zhangjinxun/Documents/Research/experiment/PreliminaryTraining/lib/SimulateDialysate/cutpredict"

# Crop all images in the folder and save them to the output folder
crop_images_in_folder(input_folder, output_folder)
