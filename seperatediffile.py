import os
import random
import shutil

# Paths to your image folders
uninfected_folder = "/Users/zhangjinxun/Documents/Research/experiment/PreliminaryTraining/lib/AllOfData450/nodarkroom/uninfected"  # Replace with your path
infected_folder = "/Users/zhangjinxun/Documents/Research/experiment/PreliminaryTraining/lib/AllOfData450/nodarkroom/infected"      # Replace with your path
train_set_folder = "/Users/zhangjinxun/Documents/Research/experiment/PreliminaryTraining/lib/training_set450"    # Replace with your desired output path

# Ensure that the train_set folder and subfolders exist
train_set_infected = os.path.join(train_set_folder, "infected")
train_set_uninfected = os.path.join(train_set_folder, "uninfected")

# Create the directories if they don't exist
os.makedirs(train_set_infected, exist_ok=True)
os.makedirs(train_set_uninfected, exist_ok=True)

# Step 1: Select all 280 images from the uninfected folder and copy them
uninfected_images = os.listdir(uninfected_folder)
for image in uninfected_images:
    image_path = os.path.join(uninfected_folder, image)
    shutil.copy(image_path, train_set_uninfected)

print(f"Copied all 280 uninfected images to {train_set_uninfected}.")

# Step 2: Randomly select 280 images from the infected folder and copy them
infected_images = os.listdir(infected_folder)
selected_infected_images = random.sample(infected_images, 280)

for image in selected_infected_images:
    image_path = os.path.join(infected_folder, image)
    shutil.copy(image_path, train_set_infected)

print(f"Randomly selected and copied 280 infected images to {train_set_infected}.")
