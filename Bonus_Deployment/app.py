# Import necessary libraries
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import os

# --- Installation Note ---
# pip install streamlit streamlit-drawable-canvas torch torchvision

# --- Define the same model architecture ---
# This is necessary for PyTorch to load the state dictionary correctly.
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

# --- Load the pre-trained model ---
@st.cache_resource
def load_trained_model():
    try:
        model = Net()
        # Look for the model in the project root
        model_path = os.path.join(os.getcwd(), 'mnist_cnn_pytorch.pth')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval() # Set the model to evaluation mode
        return model
    except (FileNotFoundError, IOError):
        st.error("Model file not found. Please make sure 'mnist_cnn_pytorch.pth' is in the project root directory and you have run the training script.")
        return None

model = load_trained_model()

# --- Streamlit App Interface ---
st.title("MNIST Handwritten Digit Classifier (PyTorch Version)")
st.write("Draw a digit (0-9) on the canvas below and click 'Predict' to see the model's guess.")

# --- Drawing Canvas ---
stroke_width = 20
stroke_color = "#FFFFFF"
bg_color = "#000000"
drawing_mode = "freedraw"

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=True,
    height=280,
    width=280,
    drawing_mode=drawing_mode,
    key="canvas",
)

# --- Prediction Logic ---
if st.button('Predict') and model is not None:
    if canvas_result.image_data is not None:
        # Get image and preprocess
        img = canvas_result.image_data.astype('uint8')
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        resized_img = cv2.resize(img_gray, (28, 28))

        # Convert to PyTorch tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        tensor = transform(resized_img).unsqueeze(0) # Add batch dimension

        # Make prediction
        with torch.no_grad():
            output = model(tensor)
        
        ps = torch.exp(output)
        probab = list(ps.numpy()[0])
        predicted_digit = probab.index(max(probab))
        confidence = max(probab)

        st.success(f"Predicted Digit: {predicted_digit}")
        st.write(f"Confidence: {confidence:.2%}")

        st.write("This is the preprocessed image sent to the model:")
        st.image(resized_img, caption="28x28 Grayscale Image", width=150)
    else:
        st.warning("Please draw a digit on the canvas before predicting.")

st.write("---")
st.write("To run this app, save the code as `app.py` and run `streamlit run app.py` in your terminal.") 