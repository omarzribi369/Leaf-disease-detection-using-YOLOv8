# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Leaf disease detection using YOLOv8-n",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Leaf disease detection using YOLOv8")

st.sidebar.header("Choose an image:")


# Sidebar
st.markdown("""**Plant Disease Detection with Deep Learning**

Using deep learning, plant diseases can be swiftly and accurately identified from images, enabling timely interventions and reducing crop losses. Deep learning models generalize well across different plant species and diseases, offering versatile and efficient solutions for agricultural sustainability. Integration of deep learning in disease detection aids in real-time monitoring, optimizing resource utilization, and enhancing global food security efforts.""")



model_path = Path(settings.DETECTION_MODEL)


# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)



source_img = None
# If image is selected
if True:
    source_img = st.sidebar.file_uploader(
        "", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Leaf detection'):
                res = model.predict(uploaded_image,
                                    conf=0.5  # Set confidence value here
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
