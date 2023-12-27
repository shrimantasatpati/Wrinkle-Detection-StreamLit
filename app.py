import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Setting page layout
st.set_page_config(
    page_title="Wrinkles segmentation using YOLOv8",
    page_icon="https://csc.edu.vn/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Wrinkles Segmentation using YOLOv8")

# Sidebar
st.sidebar.header("ML Model Config")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 5, 100, 20)) / 100
max_det = st.sidebar.slider(
    "Select maximum number of detected objects", 5, 1000, 20)
show_labels = st.sidebar.radio("Show Labels", [True, False], index=0)
show_boxes = st.sidebar.radio("Show Boxes", [True, False], index=0)

# Load YOLO model
@st.cache_resource()
def main_model():
    model = YOLO('best.pt')
    return model

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...",
                            type=['jpg', 'jpeg', 'png', "bmp", "webp"]
                            )

col1, col2 = st.columns(2)

with col1:
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image, caption="Uploaded Image",
                use_column_width=True)
    else:
        st.info("Please upload an image.")

with col2:
    if st.sidebar.button('Detect') and uploaded_file is not None:
        try:
            # Make predictions on the uploaded image
            with torch.no_grad():
                model = main_model()
                results = model.predict(
                    task="segment",
                    source=uploaded_image,
                    max_det=max_det,
                    conf=confidence,
                    show_labels=show_labels,
                    show_boxes=show_boxes,
                    save=True,
                    device="cpu"
                )
                boxes = results[0].boxes
                res_plotted = results[0].plot()[:, :, ::-1]

                # Convert the PIL image to a NumPy array
                numpy_image = np.array(res_plotted)

                # Display the segmented image
                st.image(numpy_image, caption='Segmented Image', use_column_width=True)

            with st.expander("Segmentation Results"):
                for box in boxes:
                    st.write(box.data)

        except Exception as ex:
            st.exception(ex)
    else:
        st.error("Please select a valid source type!")
