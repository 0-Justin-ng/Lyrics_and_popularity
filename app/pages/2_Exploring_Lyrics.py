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
st.write('Here is some quick information about the dataset itself.')

with open(FIG_PATH / 'summary_df.pkl', 'rb') as file:
    summary_df = joblib.load(file)
st.dataframe(summary_df)


st.write('---')


#--------------------------------------------------------------
st.image(Image.open(FIG_PATH/'4.-808s-_-Heartbreak-Album-Art.jpg'))
st.title('Distribution of Release Years')
st.write('Most of the songs in this Dataset have release years ranging from 2009 to 2019.\
         \n\n2015 seems to be the year with the most releases compared to any other year in this dataset.\
         \nHover over the bars for a closer look.')
with open(FIG_PATH / 'distribution_release_year.pkl', 'rb') as file:
    fig = joblib.load(file)

st.plotly_chart(fig, theme="streamlit")


#---------------------------------------------------------------------

st.title('Looking into the Rise of Hip Hop')
st.write('Hip hop has encountered a meteoric rise in the past four decades. We can see that \
         in the early 80\'s there is minimal representation of hip hop in the dataset, \
         but this representation sky rockets in the mid 80\'s. The peak representation occurs \
         in the dawn of the 90\'s, with approximately **80% of the songs in the dataset** being hip hop. \
         Hip hop seems to be falling in representation, but this single \
         genre remains steady representing approximately **50\% of the dataset** going into the early \
         2010\'s. Only time will tell if hip hops dominance of the charts will continue for another four decades.')

with open(FIG_PATH / 'growth_of_hip_hop.pkl', 'rb') as file:
    fig3 = joblib.load(file)

st.plotly_chart(fig3, theme="streamlit")

# ----------------------------------------------------------------
st.image(Image.open(FIG_PATH/'black and white painting (2).png'))
st.title('Most Common Words Used Overall')
st.write('Here we highlight the most common words used in the dataset. Unsurprisingly, most of these words are explicit due to a majority of the dataset containing hip hop songs. Also there seems to be a focus on individuals describing their own experiences, with "I\'m" being the most common word used in the entire dataset.')
with open(FIG_PATH / 'distribution_words_overall.pkl', 'rb') as file:
    fig2 = joblib.load(file)

st.plotly_chart(fig2, theme="streamlit")

# ----------------------------------------------------------------

st.title('Most Common Verbs Used')
st.write('Here are some of the most common verbs artists are using.')
with open(FIG_PATH / 'most_common_verb.pkl', 'rb') as file:
    fig3 = joblib.load(file)

st.plotly_chart(fig3, theme="streamlit")





# -----------------------------------------------------------------
st.image(Image.open(FIG_PATH/'t-kendrick-lamar-2018-08.webp'))
st.title("Looking at Kendrick Lamar's Favourite Words")
st.write('Kendrick Lamar is considered by some as one of the greatest hip hop artists of all time. \
         Here are some of the most common words that Kendrick Lamar has used in the 217 songs found in this dataset.')

with open(FIG_PATH / 'most_common_kendrick_lamar.pkl', 'rb') as file:
    fig3 = joblib.load(file)

st.plotly_chart(fig3, theme="streamlit")

# -----------------------------------------------------------------
st.image(Image.open(FIG_PATH/'Pink-Floyd-Dark-Side-Of-The-Moon.jpg'))
st.title("Looking at The Best Words to Use for a Popular Song")
st.write('Here are the words that increase the odds of a song being popular. \
         \n\nWe can see that most of the words have positive feelings associated with them and \
         many deal with physical contact (`feel`, `hold` and `kiss`). Additionally, certain \
         words may look strange as they were stemmed, for example `remember` was turned into \
         `rememb`. Also mentioning `la`, which was originally `LA` (transformed during the text \
         cleaning stage), seems to also increase the popularity of a song.')

with open(FIG_PATH / 'Top_20_words_for_popular_songs.pkl', 'rb') as file:
    fig3 = joblib.load(file)

st.plotly_chart(fig3, theme="streamlit")

st.title("Avoid These Words if You Want a Hit")
st.write('Here are the words that increase the odds of a song of being unpopular. \
         \n\nShockingly we can see that mentioning `Drake` in a song increases the \
         odds of the model classifying the song as an unpopular song by 2.2. This could be \
         attributed to inexperienced "Soundcloud Rappers" commenting on  `Drake` in \
         their songs. This is supported as some of the other words that increase the \
         odds of an unpopular song also are correlated to generic sounding novice \
         rapper themes such as `money`.') 

with open(FIG_PATH / 'Top_20_words_for_unpopular_songs.pkl', 'rb') as file:
    fig3 = joblib.load(file)

st.plotly_chart(fig3, theme="streamlit")