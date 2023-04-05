import pandas as pd
import numpy as np
import streamlit as st
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from utilities import utils
import joblib


st.title('Predicting Spotify Popularity')
st.write("Do you have what it takes to write a popular song? \
        \nWell, here's your chance to test your skills. \
        \nInput some lyrics below and lets see whether or not it will be a hit."
        )

MODEL_PATH = utils.get_datapath('model') / 'log_reg_spotify_popularity_pipeline.pkl'

with open(MODEL_PATH, 'rb') as file:
    pipeline = joblib.load(file)

lyric = st.text_input(label='Placeholder',label_visibility='hidden')


if lyric:
    clean_lyric = utils.clean_lyrics(lyric)
    
    clean_lyric = [clean_lyric]
    print(clean_lyric)

    result = pipeline.best_estimator_.predict(clean_lyric)
    
    print(result)
    if result == 2:
        st.write('### This song could be a hit.')
    elif result == 1:
        st.write('### This song has a medium chance of success.')
    else:
        st.write("### This song doesn't stand a chance of succeeding.")