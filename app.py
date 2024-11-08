import streamlit as st
import cv2
import os
import numpy as np
from script import load_images, resizeImage, preprocess_image, segment_image, apply_convex_hull, extract_femur_length, fetus_height

image_dir = './dataset/images'
path = './results'
images = load_images(image_dir)

st.title("Ultrasound Image Processing")

image_selection = st.selectbox("Select an image to process", range(len(images)))
selected_image = images[image_selection]

st.subheader("Original Image")
with st.container():
    col1 = st.columns(1)[0]
    col1.image(selected_image, caption="Original Image", width=512)

resized_image = resizeImage(selected_image)
preprocessed_img = preprocess_image(resized_image)

st.subheader("Preprocessing and Threshold Adjustment")


with st.container():
    threshold = st.slider("Threshold Value", min_value=0, max_value=255, value=125)
    _, thresholded_img = cv2.threshold(preprocessed_img, threshold, 255, cv2.THRESH_BINARY)
    st.image(thresholded_img, caption=f"Thresholded Image at {threshold}", width=512, )

if st.button("Process with Selected Threshold"):
    segmented_img = segment_image(thresholded_img)
    
    contours, convex_hull_img = apply_convex_hull(segmented_img)
    
    femur_length, point1, point2, marked_image = extract_femur_length(convex_hull_img, resized_image)
    marked_image = cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB)

    name = f"processed_image_{image_selection}"
    
    st.subheader("Segmented Image with Convex Hull")

    with st.container():
        st.image(convex_hull_img, caption="Convex Hull", width=512)
        
    st.subheader("Marked Image with Femur Length")

    with st.container():
        st.image(marked_image, caption="Marked Image with Femur Length", channels="RGB", width=512)
        st.write(f"Femur Length: {femur_length:.2f} pixels")
        
    if femur_length:
        
        fetus_height_cm, length_in_cm = fetus_height(femur_length)
        st.header("Results")
        col6, col7 = st.columns(2)
        
        with col6:
            st.write(f"Femur Length in cm: {length_in_cm:.2f} cm")
            st.write(f"Fetus Height: {fetus_height_cm:.2f} cm")
