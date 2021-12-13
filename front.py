import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image

def png_bytes_to_numpy(png):
    """Convert png bytes to numpy array"""
    return np.array(Image.open(BytesIO(png)))

st.title('Image to text')

uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
	bytes_data = uploaded_file.getvalue()
	img = png_bytes_to_numpy(bytes_data)
	st.image(img)

