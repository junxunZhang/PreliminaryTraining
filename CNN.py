#importing libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(40)
tf.random.set_seed(40)

# Define paths to the dataset
train_path = '/Users/zhangjinxun/Documents/Research/experiment/PreliminaryTraining/lib/SimulateDialysate/train_set'

# Data Augmentation for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,  # Increased rotation for variety
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% validation data
)

# Load the data from the folders: infectedcuted and uninfectedcuted
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

valid_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Define a more complex CNN model with more filters
model = Sequential([
    # First block
    Conv2D(64, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    # Second block with more filters
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Third block with even more filters
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Fourth block with even more filters
    Conv2D(512, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Global average pooling instead of Flatten
    GlobalAveragePooling2D(),

    # Fully connected layers
    Dense(128, activation='relu'),
    Dropout(0.4),  # Dropout to prevent overfitting
    Dense(1, activation='sigmoid')  # Binary classification output
])

# Compile the model with a reduced learning rate
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=valid_generator
)

# Fine-tune the model (optional)
# Reduce the learning rate and train for more epochs
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
history_finetune = model.fit(
    train_generator,
    epochs=10,
    validation_data=valid_generator
)

# Combine both training phases into one for plotting
history.history['accuracy'] += history_finetune.history['accuracy']
history.history['val_accuracy'] += history_finetune.history['val_accuracy']
history.history['loss'] += history_finetune.history['loss']
history.history['val_loss'] += history_finetune.history['val_loss']

# Plot the training and validation accuracy and loss
epochs = list(range(1, len(history.history['accuracy']) + 1))

# Plot accuracy
plt.plot(epochs, history.history['accuracy'], label='Train Accuracy')
plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.xticks(epochs, [f'{e:.1f}' for e in epochs])  # Show epochs with one decimal
plt.show()

# Plot loss
plt.plot(epochs, history.history['loss'], label='Train Loss')
plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.xticks(epochs, [f'{e:.1f}' for e in epochs])  # Show epochs with one decimal
plt.show()

# Evaluate the model on validation data
valid_generator.reset()  # Reset the validation generator
preds = model.predict(valid_generator)
y_pred = np.round(preds).astype(int)
y_true = valid_generator.classes

# Print the confusion matrix and classification report
print('Confusion Matrix')
print(confusion_matrix(y_true, y_pred))

print('Classification Report')
print(classification_report(y_true, y_pred))



'''
# Save the model
model.save('cnn_model.h5')

# Load the model later (if needed)
# model = tf.keras.models.load_model('cnn_model.h5')




import os
from tensorflow.keras.preprocessing import image

# Define the folder containing the images to predict
predict_folder = '/Users/zhangjinxun/Documents/Research/experiment/PreliminaryTraining/lib/SimulateDialysate/predict'



# Loop through each image in the "predict" folder
for img_file in os.listdir(predict_folder):
    if img_file.endswith('.JPG'):  # Only process .JPG files
        # Set the full path to the image
        img_path = os.path.join(predict_folder, img_file)

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(300, 300))  # Resize image to 300x300
        img_array = image.img_to_array(img)  # Convert image to numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 300, 300, 3)
        img_array /= 255.0  # Normalize the image to the range [0, 1]

        # Make a prediction
        prediction = model.predict(img_array)

        # Interpret the prediction
        if prediction[0] > 0.5:
            print(f"Prediction for {img_file}: Infected")
        else:
            print(f"Prediction for {img_file}: Uninfected")
'''
