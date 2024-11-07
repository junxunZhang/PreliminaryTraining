import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# using GPU acceleration to train the model

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing import image
import time
# Set random seed for reproducibility
np.random.seed(40)
tf.random.set_seed(40)

# Define paths to the dataset
train_val_path = '/Users/zhangjinxun/Documents/Research/experiment/PreliminaryTraining/lib/3tvt255x255/training_set'
test_path = '/Users/zhangjinxun/Documents/Research/experiment/PreliminaryTraining/lib/3tvt255x255/test_set'

# Parameters
batch_size = 32
epochs = 30  # Total epochs (20 for initial learning rate, 10 for fine-tuning)

# Define a learning rate schedule
def lr_schedule(epoch):
    return 1e-4 if epoch < 20 else 1e-5

# Add the learning rate scheduler callback
lr_scheduler = LearningRateScheduler(lr_schedule)

# Data generators
train_val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # Use 20% of the data for validation
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create train and validation data generators
train_generator = train_val_datagen.flow_from_directory(
    train_val_path,
    target_size=(255, 255),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_datagen = ImageDataGenerator(rescale=1./255)
# it is not necessary to use data augmentation for validation data

val_generator = val_datagen.flow_from_directory(
    train_val_path,
    target_size=(255, 255),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
)
#validation cannot shuffle!!!!!!, or the validation accuracy will very worst!!

# Model definition
model = Sequential([
    Input(shape=(255, 255, 3)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(512, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[lr_scheduler]  # Use the learning rate scheduler
)

# Plot training and validation accuracy
epochs_range = range(1, epochs + 1)
plt.plot(epochs_range, history.history['accuracy'], label='Train Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot training and validation loss
plt.plot(epochs_range, history.history['loss'], label='Train Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# Evaluate on Test Data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(255, 255),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy}')
print(f'Test Loss: {test_loss}')

# Save the final model
model.save('CNNModel_SingleFold.h5')


'''
# Load the model from the saved file
model = load_model('CNNModel_SingleFold.h5')

# Make predictions on new images in the "predict" folder
for img_file in os.listdir(predict_folder):
    if img_file.endswith('.JPG'):  # Only process .JPG files
        # Set the full path to the image
        img_path = os.path.join(predict_folder, img_file)

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(255, 255))  # Resize image to 255x255
        img_array = image.img_to_array(img)  # Convert image to numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 255, 255, 3)
        img_array /= 255.0  # Normalize the image to the range [0, 1]

        # Make a prediction
        prediction = model.predict(img_array)

        # Interpret the prediction
        if prediction[0] > 0.5:
            print(f"Prediction for {img_file}: Infected")
        else:
            print(f"Prediction for {img_file}: Uninfected")
'''