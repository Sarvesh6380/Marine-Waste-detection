import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Dropout, Input, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# Configuration
TRAIN_DIR = r"D:\Codes\CPP\dataset\newdataimg\train"
TEST_DIR = r"D:\Codes\CPP\dataset\newdataimg\test"
VALID_DIR = r"D:\Codes\CPP\dataset\newdataimg\valid"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 2  # Waste and Non-waste

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    print("Loading dataset...")
    
    # Load training data
    train_images_dir = os.path.join(TRAIN_DIR, 'images')
    train_labels_dir = os.path.join(TRAIN_DIR, 'labels')
    
    # Load validation data
    val_images_dir = os.path.join(VALID_DIR, 'images')
    val_labels_dir = os.path.join(VALID_DIR, 'labels')
    
    # Load test data
    test_images_dir = os.path.join(TEST_DIR, 'images')
    test_labels_dir = os.path.join(TEST_DIR, 'labels')
    
    images = []
    labels = []
    
    # Load training images and labels
    for img_name in os.listdir(train_images_dir):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            # Load image
            img_path = os.path.join(train_images_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0  # Normalize
                images.append(img)
                
                # Load corresponding label
                label_name = os.path.splitext(img_name)[0] + '.txt'
                label_path = os.path.join(train_labels_dir, label_name)
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        label = int(f.read().strip())
                        labels.append(label)
                else:
                    print(f"Warning: No label found for {img_name}")
    
    return np.array(images), np.array(labels)

def create_data_generators():
    """Create data generators for training and validation"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(TRAIN_DIR, 'images'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_directory(
        os.path.join(VALID_DIR, 'images'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    return train_generator, val_generator

def create_cnn_model():
    """Create a simple CNN model for waste detection"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

def create_rcnn_model():
    """Create an RCNN model for waste detection"""
    # Input layer
    input_layer = Input(shape=(224, 224, 3))
    
    # Feature extraction layers
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(2, 2)(x)
    
    # Region proposal network
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    
    # Classification head
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=input_layer, outputs=x)
    return model

def create_mobilenet_model():
    """Create a MobileNetV2 model for waste detection"""
    # Load the pre-trained MobileNetV2 model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom layers on top
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    return model

def train_models():
    """Train all three models and save them"""
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), 'models', 'trained_models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Train CNN model
    print("Training CNN model...")
    cnn_model = create_cnn_model()
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn_model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ]
    )
    cnn_model.save(os.path.join(models_dir, 'cnn_model.h5'))
    
    # Train RCNN model
    print("Training RCNN model...")
    rcnn_model = create_rcnn_model()
    rcnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    rcnn_model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ]
    )
    rcnn_model.save(os.path.join(models_dir, 'rcnn_model.h5'))
    
    # Train MobileNet model
    print("Training MobileNet model...")
    mobile_model = create_mobilenet_model()
    mobile_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    mobile_model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ]
    )
    mobile_model.save(os.path.join(models_dir, 'mobile_model.h5'))
    
    print("All models have been trained and saved successfully!")

if __name__ == '__main__':
    train_models() 