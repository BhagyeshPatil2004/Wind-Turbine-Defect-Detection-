import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import time
from ultralytics import YOLO  # type: ignore
import tempfile

# 1. Load the YOLOv8 model
model_path = r"runs/detect/train6/weights/best.pt"
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please check the path.")
    st.stop()
try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Defect Detection with YOLOv8 and Streamlit")

# 2. Sidebar: choose upload vs folder
st.sidebar.header("Choose Input Method")
method = st.sidebar.radio("", ["Upload Image", "Select from Folder"])

uploaded_file = None
image_path = None

if method == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        try:
            # Create a temporary directory if it doesn't exist
            temp_dir = "temp_uploaded_images"
            os.makedirs(temp_dir, exist_ok=True)

            # Sanitize the filename
            filename = os.path.basename(uploaded_file.name)
            temp_image_path = os.path.join(temp_dir, filename)

            # Save the uploaded file to the temporary directory
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            image_path = temp_image_path

            pil_img = Image.open(uploaded_file).convert("RGB")
            st.image(pil_img, caption="Uploaded Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error processing uploaded image: {e}")
            image_path = None # Ensure image_path is None if there's an error
else:
    image_folder = "Test_Img"
    if not os.path.exists(image_folder):
        st.error(f"Folder '{image_folder}' not found.")
        st.stop()
    imgs = [f for f in os.listdir(image_folder) if f.lower().endswith(("jpg","png","jpeg"))]
    if not imgs:
        st.error("No images in folder.")
        st.stop()
    sel = st.sidebar.selectbox("Choose an image", imgs)
    image_path = os.path.join(image_folder, sel)
    try:
        pil_img = Image.open(image_path).convert("RGB")
        st.image(pil_img, caption="Selected Image", use_column_width=True)
    except Exception as e:
        st.error(f"Error opening selected image: {e}")
        image_path = None # Ensure image_path is None if there's an error

# 3. Inference
if st.button("Run Inference") and image_path is not None:
    try:
        start = time.time()
        # Use the file path for inference in both upload and folder selection
        results = model(image_path, imgsz=1024)
        end = time.time()

        # plot and show
        res_img = results[0].plot(save=False)
        st.image(res_img, caption="Detection Result", use_column_width=True)
        st.success(f"Inference time: {end - start:.2f} sec")

        # Clean up the temporary file if it was created
        if method == "Upload Image" and os.path.exists(image_path) and "temp_uploaded_images" in image_path:
            os.remove(image_path)

    except Exception as e:
        st.error(f"Error during inference: {e}")
elif st.button("Run Inference") and image_path is None:
    st.warning("Please upload or select an image before running inference.")
