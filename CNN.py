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
train_val_path = '/Users/zhangjinxun/Documents/Research/experiment/PreliminaryTraining/lib/3tvt255x255/training_set'
test_path = '/Users/zhangjinxun/Documents/Research/experiment/PreliminaryTraining/lib/3tvt255x255/test_set'

# Data Augmentation for training and validation (from training_set)
train_val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # Use 20% of training_set for validation
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest'
)

# No augmentation for test set, only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the training and validation data (split from training_set)
train_generator = train_val_datagen.flow_from_directory(
    train_val_path,
    target_size=(255, 255),
    batch_size=32,
    class_mode='binary',
    subset='training'  # Training subset
)

valid_generator = train_val_datagen.flow_from_directory(
    train_val_path,
    target_size=(255, 255),
    batch_size=32,
    class_mode='binary',
    subset='validation'  # Validation subset
)

# Load test data
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(255, 255),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Model definition (unchanged)
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(255, 255, 3)),
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

# Compile the model (unchanged)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=valid_generator
)

# Fine-tune the model (unchanged)
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
history_finetune = model.fit(
    train_generator,
    epochs=10,
    validation_data=valid_generator
)

# Combine both training phases for plotting
history.history['accuracy'] += history_finetune.history['accuracy']
history.history['val_accuracy'] += history_finetune.history['val_accuracy']
history.history['loss'] += history_finetune.history['loss']
history.history['val_loss'] += history_finetune.history['val_loss']

# Plot accuracy
epochs = list(range(1, len(history.history['accuracy']) + 1))
plt.plot(epochs, history.history['accuracy'], label='Train Accuracy')
plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.plot(epochs, history.history['loss'], label='Train Loss')
plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
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
model.save('CNNModel101824.h5')






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
