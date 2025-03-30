import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your model
model = load_model('best_model.h5')

# Define a function to predict the traffic sign
def predict_sign(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    probability = np.max(predictions[0])
    
    # Map predicted class to sign name (adjust this dictionary as needed)
    sign_names = {0: "Stop", 1: "Yield", 2: "Speed Limit", 3: "No Entry"}
    sign_name = sign_names.get(predicted_class, "Unknown")
    
    return predicted_class, sign_name, probability

# Function to open file dialog and make prediction
def upload_and_predict():
    img_path = filedialog.askopenfilename()
    pred_class, sign_name, prob = predict_sign(img_path)
    result_label.config(text=f"Predicted: {sign_name} (Class: {pred_class}, Probability: {prob:.3f})")

# Create the GUI
root = tk.Tk()
root.title("Traffic Sign Predictor")

upload_button = tk.Button(root, text="Upload Image", command=upload_and_predict)
upload_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()