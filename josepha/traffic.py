import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split

EPOCHS = 15
IMG_WIDTH = 32
IMG_HEIGHT = 32
NUM_CATEGORIES = 43
TEST_SIZE = 0.2

def main():
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    images, labels = load_data(sys.argv[1])

    if len(images) == 0:
        sys.exit("Error: No images found in the specified directory.")

    labels = tf.keras.utils.to_categorical(labels, num_classes=NUM_CATEGORIES)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE, random_state=42
    )

    model = get_model()
    model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test), verbose=1)

    filename = sys.argv[2] if len(sys.argv) == 3 else 'best_model.h5'
    model.save(filename)
    print(f"Model saved to {filename}.")

def load_data(data_dir):
    images = []
    labels = []
    
    for category in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(category))
        
        if not os.path.isdir(category_dir):
            print(f"Warning: Directory {category_dir} does not exist.")
            continue
            
        for image_file in os.listdir(category_dir):
            if image_file.endswith('.ppm'):  # Check for .ppm files
                image_path = os.path.join(category_dir, image_file)
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Warning: Could not read image {image_path}. Skipping.")
                        continue
                    
                    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = image / 255.0
                    
                    images.append(image)
                    labels.append(category)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue
                
    return (images, labels)
    images = []
    labels = []
    
    print("Checking contents of:", data_dir)
    print(os.listdir(data_dir))  # Print the contents of the directory
    
    for category in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(category))
        
        if not os.path.isdir(category_dir):
            print(f"Warning: Directory {category_dir} does not exist.")
            continue
            
        for image_file in os.listdir(category_dir):
            if image_file.endswith('.ppm'):  # Check for .ppm files
                image_path = os.path.join(category_dir, image_file)
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Warning: Could not read image {image_path}. Skipping.")
                        continue
                    
                    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = image / 255.0
                    
                    images.append(image)
                    labels.append(category)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue
                
    return (images, labels)
def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

if __name__ == "__main__":
    main()