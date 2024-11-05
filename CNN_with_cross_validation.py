import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing import image

# Set random seed for reproducibility
np.random.seed(40)
tf.random.set_seed(40)

# Define paths to the dataset
train_val_path = '/Users/zhangjinxun/Documents/Research/experiment/PreliminaryTraining/lib/3tvt255x255/training_set'
test_path = '/Users/zhangjinxun/Documents/Research/experiment/PreliminaryTraining/lib/3tvt255x255/test_set'

# Parameters
num_folds = 5  # Number of folds
batch_size = 32
epochs = 30  # Total epochs (20 for initial learning rate, 10 for fine-tuning)

# Prepare arrays to store metrics for each fold
accuracy_per_fold = []
loss_per_fold = []
histories = []

# Get list of image paths and labels
data_generator = ImageDataGenerator(rescale=1./255)
train_data = data_generator.flow_from_directory(
    train_val_path,
    target_size=(255, 255),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

# Extract image file paths and corresponding labels
file_paths = train_data.filepaths
labels = train_data.classes

# Define k-fold cross-validator
kf = KFold(n_splits=num_folds, shuffle=True, random_state=40)

# Define a learning rate schedule
def lr_schedule(epoch):
    return 1e-3 if epoch < 20 else 1e-5

# Add the learning rate scheduler callback
lr_scheduler = LearningRateScheduler(lr_schedule)

fold_no = 1
for train_index, val_index in kf.split(file_paths):
    print(f'\nTraining for fold {fold_no}...\n')
    
    # Split data for this fold
    train_files = [file_paths[i] for i in train_index]
    train_labels = [str(labels[i]) for i in train_index]
    val_files = [file_paths[i] for i in val_index]
    val_labels = [str(labels[i]) for i in val_index]

    # Data generators for each fold
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        brightness_range=[0.7, 1.3],
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': train_files, 'class': train_labels}),
        x_col='filename',
        y_col='class',
        target_size=(255, 255),
        batch_size=batch_size,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': val_files, 'class': val_labels}),
        x_col='filename',
        y_col='class',
        target_size=(255, 255),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

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
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model for the current fold
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[lr_scheduler]  # Use the learning rate scheduler
    )
    
    # Append history for later plotting
    histories.append(history)
    
    # Save accuracy and loss for this fold
    accuracy_per_fold.append(history.history['val_accuracy'][-1])
    loss_per_fold.append(history.history['val_loss'][-1])

    # Delay to avoid log overlap for the next fold
    time.sleep(1)

    # Increment fold number
    fold_no += 1

# Plot accuracy and loss for each fold
for i, history in enumerate(histories):
    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, history.history['accuracy'], label=f'Train Accuracy Fold {i+1}')
    plt.plot(epochs_range, history.history['val_accuracy'], label=f'Validation Accuracy Fold {i+1}')

plt.title('Model Accuracy per Fold')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

for i, history in enumerate(histories):
    plt.plot(epochs_range, history.history['loss'], label=f'Train Loss Fold {i+1}')
    plt.plot(epochs_range, history.history['val_loss'], label=f'Validation Loss Fold {i+1}')

plt.title('Model Loss per Fold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# Print cross-validation results
print('Cross-validation results:')
print(f'Average Validation Accuracy: {np.mean(accuracy_per_fold):.2f} (+/- {np.std(accuracy_per_fold):.2f})')
print(f'Average Validation Loss: {np.mean(loss_per_fold):.2f} (+/- {np.std(loss_per_fold):.2f})')

# Evaluate on Test Data if required
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
model.save('CNNModel_WithCrossValidation.h5')


'''
# Load the model from the saved file
model = load_model('CNNModel_CrossValidation.h5')

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
