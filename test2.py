from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import streamlit as st

def teachable_machine_classification(img, model_path, labels_path):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model(model_path, compile=False)

    # Load the labels
    class_names = open(labels_path, "r").readlines()

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Resize the image to be at least 224x224 and then crop from the center
    size = (224, 224)
    image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name.strip(), confidence_score

st.set_page_config(layout="wide")

st.title("Test Drawings for Parkinson's")

# File uploaders for three images
input_img_clock = st.file_uploader("Enter your Clock drawing", type=['jpeg', 'jpg', 'png'])
input_img_spiral = st.file_uploader("Enter your Spiral drawing", type=['jpeg', 'jpg', 'png'])
input_img_wave = st.file_uploader("Enter your Wave drawing", type=['jpeg', 'jpg', 'png'])

# Model and labels paths for three models
model_paths = ["keras_model_clock.h5", "keras_model_spiral.h5", "keras_model_wave.h5"]
labels_paths = ["labels_clock.txt", "labels_spiral.txt", "labels_wave.txt"]

if input_img_clock is not None and input_img_spiral is not None and input_img_wave is not None:
    if st.button("Test"):
        col1, col2, col3 = st.columns(3)

        # Process and display results for the first image
        with col1:
            st.info("Your uploaded Clock drawing")
            st.image(input_img_clock, use_column_width=True)
            img_clock = Image.open(input_img_clock).convert("RGB")
            # Process image 1 with model 1
            class_name1, confidence_score1 = teachable_machine_classification(img_clock, model_paths[0], labels_paths[0])
            st.write(f"Class: {class_name1}")
            st.write(f"Confidence Score: {confidence_score1:.2f}")

        # Process and display results for the second image
        with col2:
            st.info("Your uploaded Spiral drawing")
            st.image(input_img_spiral, use_column_width=True)
            img_spiral = Image.open(input_img_spiral).convert("RGB")
            # Process image 2 with model 2
            class_name2, confidence_score2 = teachable_machine_classification(img_spiral, model_paths[1], labels_paths[1])
            st.write(f"Class: {class_name2}")
            st.write(f"Confidence Score: {confidence_score2:.2f}")

        # Process and display results for the third image
        with col3:
            st.info("Your uploaded Wave drawing")
            st.image(input_img_wave, use_column_width=True)
            img_wave = Image.open(input_img_wave).convert("RGB")
            # Process image 3 with model 3
            class_name3, confidence_score3 = teachable_machine_classification(img_wave, model_paths[2], labels_paths[2])
            st.write(f"Class: {class_name3}")
            st.write(f"Confidence Score: {confidence_score3:.2f}")
