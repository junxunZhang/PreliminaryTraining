# PreliminaryTraining
# 2024-10-09 Experiment Report

## 1. Experiment Overview
- **Model**: Convolutional Neural Network (CNN) for binary classification of peritoneal dialysate images.
- **Objective**: Train a CNN to distinguish between infected and uninfected peritoneal dialysate.
- **Dataset**: 
  - **Training set**: 2848 images (split 80/20 for training and validation).
  - **Validation set**: 712 images.

## 2. Model Architecture
- **Input**: Images of size 200x200 with 3 channels (RGB).
- **Layers**:
  1. Four convolutional blocks, each followed by Batch Normalization and MaxPooling.
  2. Global Average Pooling for dimensionality reduction.
  3. Two Dense layers (128 units with ReLU and a binary output layer with sigmoid activation).
  4. Dropout (0.5) for regularization.
- **Optimizer**: Adam (learning rate 1e-4 for initial training, 1e-5 for fine-tuning).
- **Loss Function**: Binary crossentropy.
- **Metrics**: Accuracy.
- ![image](https://github.com/user-attachments/assets/9cabad2a-3741-4f14-8aba-45c76605aa68)
- ![image](https://github.com/user-attachments/assets/efb9dc03-7a45-45cc-9b67-a5834fd88ae0)
- ![image](https://github.com/user-attachments/assets/b4f692bc-8d2a-4d3b-b35d-aa1f2ff90294)


## 3. Training Process
### Part 1: Initial Training (20 Epochs)
- **Data Augmentation**:
  - **Rescale**: Normalize image pixel values by dividing by 255.
  - **Transformation**: Rotation, shift, shear, zoom, brightness, and horizontal flip for robustness.
- **Training Summary**:
  - **Initial learning rate**: \(1 \times 10^{-4}\).
  - **Epoch 1-6**: Validation accuracy fluctuated around 50-55%, indicating that the model struggled initially to generalize.
  - **Epoch 7-10**: A significant boost in performance, with validation accuracy improving to around 88-90%.
  - **Epoch 11-20**: Validation accuracy stabilized between 85-90%, with loss decreasing, but some minor fluctuations in validation loss were observed, potentially indicating slight overfitting.

### Part 2: Fine-Tuning (10 Additional Epochs)
- **Purpose**: Fine-tune the model to achieve higher generalization.
- **Learning Rate**: Lowered to \(1 \times 10^{-5}\).
- **Result**:
  - **Improvement in validation accuracy**: Rose to over 90% (peaking at ~95% by the end of fine-tuning).
  - **Consistent decrease in validation loss**, indicating that the model was better generalizing to the validation set.

## 4. Training & Validation Accuracy and Loss
### Training Accuracy:
- **Trend**: Smooth and steady increase in training accuracy, approaching ~90% by epoch 20.
- **Conclusion**: The model consistently learned the training data.
  
### Validation Accuracy:
- **Initial instability**: Validation accuracy was erratic during the first few epochs, stuck around 50%, which suggests the model had difficulty with generalizing early on.
- **Later improvement**: By epoch 7, validation accuracy sharply improved and stabilized at around 88-90%.

### Loss:
- **Training loss**: Smooth decline as expected, indicating good learning progress.
- **Validation loss**: Initially high and unstable, but reduced significantly after epoch 6, aligning well with the accuracy trends.

## 5. Confusion Matrix & Classification Report
- **Confusion Matrix**:
  - Class 0 (uninfected): 49% precision and recall.
  - Class 1 (infected): 47% precision and recall.
  
- **Classification Report**:
  - **Accuracy**: 48% overall.
  - **Macro avg/Weighted avg**: Both at 48% for precision, recall, and F1-score.
  - **Interpretation**: The model struggles with distinguishing between the two classes with high precision and recall, possibly due to slight visual differences between infected and uninfected images.

## 6. Model Training Strategy (Two-Part Training)
- **Initial Training** (Epochs 1-20):
  - The model was trained with a relatively higher learning rate (\(1 \times 10^{-4}\)) to allow for faster convergence. However, the initial stages were unstable, as seen in the validation accuracy. This indicates that the model needed more refined learning updates to generalize better.
  
- **Fine-Tuning** (Epochs 21-30):
  - Lowering the learning rate (\(1 \times 10^{-5}\)) in the fine-tuning phase allowed the model to make smaller, more precise adjustments. This is a common practice to help the model avoid overfitting or getting stuck in local minima while ensuring that the model achieves a higher generalization capability.

## 7. Potential Issues and Improvements
- **Initial Training Instability**: Early validation accuracy stagnation might indicate that the model struggled with feature extraction. Using transfer learning from a pre-trained model might help.
- **Overfitting Risk**: The small gap between training and validation performance suggests mild overfitting. Strategies like early stopping or further regularization (Dropout, L2 regularization) could be beneficial.
- **Class Imbalance**: The low precision/recall for both classes suggests that the model struggles to differentiate between them. Addressing the dataset's inherent difficulty (e.g., visual similarities) or using techniques like focal loss might improve performance.
