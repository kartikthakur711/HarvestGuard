import streamlit as st
from PIL import Image
from predict import load_classifier, predict

st.set_page_config(page_title="HarvestGuard", page_icon="🌱")
st.title("HarvestGuard")
st.write("Upload a plant leaf image to view the top three model predictions.")
st.caption("Research prototype. Field-image performance is unverified; predictions are not a diagnosis.")

@st.cache_resource
def classifier():
    return load_classifier()

upload = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])
if upload:
    try:
        image = Image.open(upload).convert("RGB")
        st.image(image, caption="Uploaded leaf")
        model, metadata = classifier()
        for label, score in predict(image, model, metadata):
            st.write(f"{label}: {score:.2%}")
    except Exception as exc:
        st.error(f"Prediction unavailable: {exc}")
