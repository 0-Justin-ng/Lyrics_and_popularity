import pandas as pd
import numpy as np
import streamlit as st
import sys
from pathlib import Path
import os

sys.path.append(Path('app/lyrics_and_popularity'))


from utilities import utils
import joblib

# Setup the model-------------------------------------------------------------------------
# This is the specific path for the streamlit app.
CURRENT_WORKING_DIRECTORY = Path(os.getcwd())
MODEL_PATH =  CURRENT_WORKING_DIRECTORY / 'model' / 'log_reg_tfidf.pkl'


with open(MODEL_PATH, 'rb') as file:
    model = joblib.load(file)

@st.cache_data()
def predicit_lyric_popularity(lyric):
    clean_lyric = utils.clean_lyrics(lyric)
    clean_lyric = utils.stop_word_removal_and_stem(clean_lyric)
    
    clean_lyric = [clean_lyric]
    result = model.best_estimator_.predict(clean_lyric)
    
    return result


# App output---------------------------------------------------------------------------
st.title('Predicting Spotify Popularity')
st.write("Do you have what it takes to write a popular song? \
        \nWell, here's your chance to test your skills. \
        \nInput some lyrics below and lets see whether or not it will be a hit."
        )

lyric = st.text_area(label='Placeholder', label_visibility='hidden')

# If the text input is not empty, then predict the lyric
if lyric:
    result = predicit_lyric_popularity(lyric)
    if result == 2:
        st.write('### 😁 This song could be a hit.')
    elif result == 1:
        st.write('### 😐 This song has a medium chance of success.')
    else:
        st.write("### 😞 This song doesn't stand a chance of succeeding.")
        st.write('Try using these words:')
        st.write('- Feel')
        st.write('- Remember')
        st.write('- Hold')
        st.write('- Drunk')     
        st.write('- Need')
        st.write('- Kiss')