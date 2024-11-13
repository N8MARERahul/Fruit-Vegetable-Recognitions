import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Set the page configuration
st.set_page_config(page_title="Calorie Calculation of Fruits & Vegetables", layout="centered")

# Page title
st.title("Calorie Calculation Of Fruits & Vegetables")

# Instructions and main sections
st.subheader("Scan your next object")

# Webcam capture section
st.write("### Scan your object here")
img_file_buffer = st.camera_input()

# Check if an image is uploaded or captured
if img_file_buffer is not None:
    # Load image from buffer and convert to OpenCV format
    image = Image.open(img_file_buffer)
    image = np.array(image)

    # Display the captured image (for demonstration purposes)
    st.image(image, caption="Captured Object", use_column_width=True)

    # Example function to calculate calories (replace with actual model)
    def calculate_calories(image):
        # Placeholder for object detection and calorie calculation logic
        return {"Object": "Carrot", "Calories per 10g": 4}

    # Perform calorie calculation
    result = calculate_calories(image)
    
    # Display the result
    st.write("### Object Details")
    st.write("**Name of the Object**:", result["Object"])
    st.write("**Calories Calculated per 10 gm**:", result["Calories per 10g"], "kcal")

    # Add spacing for layout aesthetics
    st.markdown("---")
else:
    st.write("Please scan an object to calculate calories.")
