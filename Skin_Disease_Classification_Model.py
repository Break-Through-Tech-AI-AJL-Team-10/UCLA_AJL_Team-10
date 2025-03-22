# 1. Import Necessary Libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 2. Load Data
train_df = pd.read_csv('/kaggle/input/bttai-ajl-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/bttai-ajl-2025/test.csv')

# Add .jpg extension to md5hash column to reference the file_name
train_df['md5hash'] = train_df['md5hash'].astype(str) + '.jpg'
test_df['md5hash'] = test_df['md5hash'].astype(str) + '.jpg'

# Combine label and md5hash to form the correct path for training images
train_df['file_path'] = train_df['label'] + '/' + train_df['md5hash']

# 3. Data Preprocessing
# Encode the labels
label_encoder = LabelEncoder()
train_df['encoded_label'] = label_encoder.fit_transform(train_df['label'])

# Split the data into training and validation sets
train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)

# 4. Define Image Data Generators for Training and Validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

# Directory paths for training data
train_dir = '/kaggle/input/bttai-ajl-2025/train/train/'

def create_generator(dataframe, directory, batch_size=32, target_size=(128, 128)):
    generator = train_datagen.flow_from_dataframe(
        dataframe=dataframe,
        directory=directory,
        x_col='file_path', 
        y_col='encoded_label',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='raw',  
        validate_filenames=False  
    )
    return generator

# Create generators for training and validation
train_generator = create_generator(train_data, train_dir)
val_generator = create_generator(val_data, train_dir)

# 5. Build the CNN Model
# Determine number of classes
num_classes = len(label_encoder.classes_)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  
])

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',  
              metrics=['accuracy'])

# 6. Train the Model
# Use EarlyStopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    train_generator,
    epochs=30,  
    validation_data=val_generator,
    callbacks=[early_stop]
)

val_loss, val_acc = model.evaluate(val_generator, steps=len(val_generator))
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

# 7. Preprocess Test Data and Create Test Generator
def preprocess_test_data(test_df, directory, batch_size=32, target_size=(128, 128)):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=directory,
        x_col='md5hash',  
        y_col=None,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,  
        validate_filenames=False
    )
    return test_generator

# Directory path for test images
test_dir = '/kaggle/input/bttai-ajl-2025/test/test/'

# Create test generator
test_generator = preprocess_test_data(test_df, test_dir)

# 8. Make Predictions on Test Data
predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

# Map the predicted class indices back to original labels
predicted_labels = label_encoder.inverse_transform(predicted_classes)

results_df = pd.DataFrame({
    'file': test_df['md5hash'],
    'predicted_label': predicted_labels
})

print(results_df.head())

# Map the predicted class indices back to original labels
predicted_labels = label_encoder.inverse_transform(predicted_classes)

# Remove the '.jpg' extension from the md5hash column for the submission file
submission_df = pd.DataFrame({
    'md5hash': test_df['md5hash'].str.replace('.jpg', '', regex=False),
    'label': predicted_labels
})

# Save to CSV
submission_df.to_csv('submission.csv', index=False)
