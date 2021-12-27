import json

import cv2
import streamlit as st
import torch

from config import IMAGE_DIM, MODEL_FILE, RAMP_FRAMES, WORD_MAP_FILE
from utils import get_frame, run_one_image


def main():
    st.set_page_config(
        page_title="Image Captioning",
        page_icon="brain.png",
        layout="wide",
        initial_sidebar_state="auto",
    )

    st.title("MSc Data Science For Business - Deep Learning 2 Project")
    st.header("Image Captioning")
    st.text(
        "Mathis BATOUL – Jocelin BORDET"
    )
    st.text("")

    device = torch.device("cpu")
    checkpoint = torch.load(MODEL_FILE, map_location=str(device))
    decoder = checkpoint["decoder"]
    decoder = decoder.to(device)
    decoder.eval()

    encoder = checkpoint["encoder"]
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(WORD_MAP_FILE, "r") as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Take picture with webcam"):
            cap = cv2.VideoCapture(0)
            cap.set(3, IMAGE_DIM)
            cap.set(4, IMAGE_DIM)

            for _ in range(RAMP_FRAMES):
                _ = cap.read()

            frame = get_frame(cap)

            st.subheader("Picture taken with webcam")
            st.image(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, img_buff_arr = cv2.imencode(".png", frame)
            img_bytes = img_buff_arr.tobytes()

            run_one_image(
                img_bytes,
                encoder,
                decoder,
                rev_word_map,
                word_map,
            )

            del cap

    with col2:
        uploaded_file = st.file_uploader("Upload an image from disk")
        if uploaded_file is not None:
            st.subheader("Uploaded file")
            st.image(uploaded_file)

            img_bytes = uploaded_file.getvalue()

            run_one_image(
                img_bytes,
                encoder,
                decoder,
                rev_word_map,
                word_map,
            )


if __name__ == "__main__":
    main()
