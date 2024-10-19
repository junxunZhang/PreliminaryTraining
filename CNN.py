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

# Define paths to the new dataset (3tvt255x255 folder)
train_path = '/Users/zhangjinxun/Documents/Research/experiment/PreliminaryTraining/lib/3tvt255x255/training_set'
valid_path = '/Users/zhangjinxun/Documents/Research/experiment/PreliminaryTraining/lib/3tvt255x255/validation_set'
test_path = '/Users/zhangjinxun/Documents/Research/experiment/PreliminaryTraining/lib/3tvt255x255/test_set'

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
    fill_mode='nearest'
)

# No data augmentation for the validation and test sets, only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the data from the folders: infected and uninfected for training
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(255, 255),
    batch_size=32,
    class_mode='binary'
)

# Load validation data
valid_generator = test_datagen.flow_from_directory(
    valid_path,
    target_size=(255, 255),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # must be False to get the correct order of predictions
)

# Load test data
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(255, 255),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Define a more complex CNN model with more filters
model = Sequential([
    # First block
    Conv2D(64, (3, 3), activation='relu', input_shape=(255, 255, 3)),
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
plt.xticks(epochs)
plt.show()

# Plot loss
plt.plot(epochs, history.history['loss'], label='Train Loss')
plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.xticks(epochs)
plt.show()

# Evaluate the model on test data
test_generator.reset()  # Reset the test generator
preds = model.predict(test_generator)
y_pred = np.round(preds).astype(int)
y_true = test_generator.classes

# Print the confusion matrix and classification report
print('Confusion Matrix')
print(confusion_matrix(y_true, y_pred))

print('Classification Report')
print(classification_report(y_true, y_pred))

# Save the model
model.save('CNNModel101824')





# Load the model later (if needed)
# model = tf.keras.models.load_model('CNNModel101324')




'''
# Load the model from the saved file
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('cnn_model.h5')

# Make predictions with the loaded model
preds = model.predict(valid_generator)
'''



'''
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
