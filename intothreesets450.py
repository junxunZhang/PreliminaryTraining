import os
import random
import shutil

# Paths to the original dataset
training_set450 = "/Users/zhangjinxun/Documents/Research/experiment/PreliminaryTraining/lib/training_set450"  # Replace with your path
tvt_folder = "/Users/zhangjinxun/Documents/Research/experiment/PreliminaryTraining/lib/tvt"  # Replace with your desired output path

# Create the "tvt" folder and the subfolders for train, validation, and test sets
train_folder = os.path.join(tvt_folder, "training_set")
val_folder = os.path.join(tvt_folder, "validation_set")
test_folder = os.path.join(tvt_folder, "test_set")

# Create infected and uninfected folders in each of the sets
for subset in [train_folder, val_folder, test_folder]:
    os.makedirs(os.path.join(subset, "infected"), exist_ok=True)
    os.makedirs(os.path.join(subset, "uninfected"), exist_ok=True)

# Function to split the images and copy them to the respective folders
def split_and_copy(images, infected, train_split, val_split, test_split):
    random.shuffle(images)  # Shuffle the images randomly

    # Calculate the number of images for each set
    num_train = int(len(images) * train_split)
    num_val = int(len(images) * val_split)

    train_images = images[:num_train]
    val_images = images[num_train:num_train + num_val]
    test_images = images[num_train + num_val:]

    # Copy images to the respective folders
    folder_type = "infected" if infected else "uninfected"
    for image in train_images:
        shutil.copy(image, os.path.join(train_folder, folder_type))
    for image in val_images:
        shutil.copy(image, os.path.join(val_folder, folder_type))
    for image in test_images:
        shutil.copy(image, os.path.join(test_folder, folder_type))

# Process infected images
infected_images = [os.path.join(training_set450, "infected", img) for img in os.listdir(os.path.join(training_set450, "infected")) if img.endswith((".JPG", ".jpeg", ".png"))]
split_and_copy(infected_images, infected=True, train_split=0.7, val_split=0.2, test_split=0.1)

# Process uninfected images
uninfected_images = [os.path.join(training_set450, "uninfected", img) for img in os.listdir(os.path.join(training_set450, "uninfected")) if img.endswith((".JPG", ".jpeg", ".png"))]
split_and_copy(uninfected_images, infected=False, train_split=0.7, val_split=0.2, test_split=0.1)

print("Data successfully split into training, validation, and test sets!")
