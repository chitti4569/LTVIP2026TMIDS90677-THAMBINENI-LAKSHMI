import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# ✅ PATHS - Ensure these are correct for your machine
data_dir = r'D:\PythonProject\static\uploads\dataset2-master\dataset2-master\images\TRAIN'
class_labels = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

filepaths, labels = [], []
for label in class_labels:
    class_dir = os.path.join(data_dir, label)
    if os.path.exists(class_dir):
        # Increased limit to 400 for better accuracy (or remove [:400] to use all)
        files = os.listdir(class_dir)[:400]
        for file in files:
            if file.lower().endswith(('.jpeg', '.png', '.jpg')):
                filepaths.append(os.path.join(class_dir, file))
                labels.append(label)

df = pd.DataFrame({'filepaths': filepaths, 'labels': labels})
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['labels'])

# ✅ IMPROVED PREPROCESSING (Added Augmentation)
# Augmentation helps the model "see" the cells from different angles/scales
datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

train_gen = datagen.flow_from_dataframe(
    train_df, x_col='filepaths', y_col='labels',
    target_size=(224, 224), batch_size=16, class_mode='categorical',
    classes=class_labels, shuffle=True
)

val_gen = datagen.flow_from_dataframe(
    val_df, x_col='filepaths', y_col='labels',
    target_size=(224, 224), batch_size=16, class_mode='categorical',
    classes=class_labels, shuffle=False
)

# ✅ TRANSFER LEARNING
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False # Freeze the base

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2), # Prevents overfitting
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ TRAIN (Increased to 10 epochs for actual learning)
print("🚀 Training started...")
model.fit(train_gen, validation_data=val_gen, epochs=10)

model.save("BloodCellModel.h5")
print("✅ Model saved as BloodCellModel.h5")