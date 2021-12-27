from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from config import BEAM_SIZE
from main import caption_image_beam_search, visualize_attention


def png_bytes_to_numpy(png):
    """Convert png bytes to numpy array"""

    return np.array(Image.open(BytesIO(png)))


def get_frame(cap):
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame


def run_one_image(
    img_bytes,
    encoder,
    decoder,
    rev_word_map,
    word_map,
    beam_size=BEAM_SIZE,
):
    img_arr = png_bytes_to_numpy(img_bytes)

    seq, alphas = caption_image_beam_search(
        encoder, decoder, img_arr, word_map, beam_size
    )
    sentence = " ".join(rev_word_map[ind] for ind in seq[1:-1])

    st.subheader("Caption")
    st.write(sentence.capitalize())

    visualize_attention(
        img_arr,
        seq,
        alphas,
        rev_word_map,
    )

    st.subheader("Attention visualization")
    st.image("./attention.png")
