import pandas as pd
import numpy as np
import streamlit as st
import sys
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent.parent))

from utilities import utils
import joblib
FIG_PATH = utils.get_datapath('figures')

import streamlit as st


st.title("Let's Explore Some Lyrics")
st.write('Here you will get a closer look into the lyrics of your favourite artists and songs.')
 
 
# --------------------------------------------------------------
st.write('---')

st.title('Summary of the Dataset')
st.write('Here is some information about the dataset itself.')

with open(FIG_PATH / 'summary_df.pkl', 'rb') as file:
    summary_df = joblib.load(file)
st.dataframe(summary_df)

st.write('---')


#--------------------------------------------------------------
st.title('Distribution of Release Years')
st.write('Most of the songs in this Dataset have release years ranging from 2009 to 2019.\
         \n2015 seems to be the year with the most releases compared to any other year in this dataset.\
         \nHover over the bars for a closer look.')
with open(FIG_PATH / 'distribution_release_year.pkl', 'rb') as file:
    fig = joblib.load(file)

st.plotly_chart(fig, theme="streamlit")



# ----------------------------------------------------------------
st.image(Image.open(FIG_PATH/'black and white painting (2).png'))
st.title('Most Common Words Used Overall')
st.write('Here we highlight the most common words used in the dataset. Unsurprisingly, most of these words are explicit due to a majority of the dataset containing hip hop songs. Also there seems to be a focus on individuals describing their own experiences, with "I\'m" being the most common word used in the entire dataset.')
with open(FIG_PATH / 'distribution_words_overall.pkl', 'rb') as file:
    fig2 = joblib.load(file)

st.plotly_chart(fig2, theme="streamlit")

# ----------------------------------------------------------------
st.image(Image.open(FIG_PATH/'4.-808s-_-Heartbreak-Album-Art.jpg'))
st.title('Most Common Verbs Used')
with open(FIG_PATH / 'most_common_verb.pkl', 'rb') as file:
    fig3 = joblib.load(file)

st.plotly_chart(fig3, theme="streamlit")

# -----------------------------------------------------------------
st.image(Image.open(FIG_PATH/'t-kendrick-lamar-2018-08.webp'))
st.title("Looking at Kendrick Lamar's Favourite Words")
st.write('Kendrick Lamar is considered by some as one of the greatest hip hop artists of all time. \
         Here are some of the most common words that Kendrick Lamar has used in the 218 songs found in this dataset.')

with open(FIG_PATH / 'most_common_kendrick_lamar.pkl', 'rb') as file:
    fig3 = joblib.load(file)

st.plotly_chart(fig3, theme="streamlit")