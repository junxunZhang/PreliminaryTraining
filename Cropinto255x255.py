import os
import cv2

# Paths to your folders
input_folder = "/Users/zhangjinxun/Documents/Research/experiment/PreliminaryTraining/lib/2tvt450x680"  # Replace with your "2tvt450x680" path
output_folder = "/Users/zhangjinxun/Documents/Research/experiment/PreliminaryTraining/lib/3tvt255x255"  # Replace with your desired "3tvt255x255" path

# Create the output folder and subfolders for training, validation, and test sets
for subset in ["training_set", "validation_set", "test_set"]:
    for folder_type in ["infected", "uninfected"]:
        os.makedirs(os.path.join(output_folder, subset, folder_type), exist_ok=True)

# Function to crop an image into 35 255x255 segments
def crop_image_into_segments(image, output_folder, base_name):
    height, width = image.shape[:2]
    
    # Set the crop size to 255x255
    crop_size = 255
    step_x = (width - crop_size) // 4  # 4 steps in the width direction
    step_y = (height - crop_size) // 6  # 6 steps in the height direction

    count = 1
    for y in range(0, step_y * 6 + 1, step_y):  # 6 steps in height
        for x in range(0, step_x * 4 + 1, step_x):  # 4 steps in width
            cropped_img = image[y:y + crop_size, x:x + crop_size]
            output_image_path = os.path.join(output_folder, f"{base_name}_crop_{count}.jpg")
            cv2.imwrite(output_image_path, cropped_img)
            count += 1

# Function to process each folder (train, validation, test, and their subfolders)
def process_folder(input_folder, output_folder):
    # Iterate through each subset (train, validation, test)
    for subset in os.listdir(input_folder):
        subset_path = os.path.join(input_folder, subset)
        
        if os.path.isdir(subset_path):
            for folder_type in ["infected", "uninfected"]:
                # Define the full input path
                folder_path = os.path.join(subset_path, folder_type)

                # Define the output path for the cropped images
                output_type_folder = os.path.join(output_folder, subset, folder_type)

                # Process each image in the folder
                for filename in os.listdir(folder_path):
                    if filename.endswith(".JPG") or filename.endswith(".jpeg") or filename.endswith(".png"):
                        # Construct the full input file path
                        input_path = os.path.join(folder_path, filename)

                        # Load the image
                        image = cv2.imread(input_path)

                        if image is not None:
                            # Crop the image and save the 255x255 segments
                            base_name = os.path.splitext(filename)[0]
                            crop_image_into_segments(image, output_type_folder, base_name)
                        else:
                            print(f"Failed to load image: {input_path}")

# Process the 2tvt450x680 folder and crop the images
process_folder(input_folder, output_folder)

print("Cropping completed successfully!")
