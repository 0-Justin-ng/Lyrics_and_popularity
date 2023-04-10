import pandas as pd
import numpy as np
import streamlit as st
import sys
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

from utilities import utils
import joblib
FIG_PATH = utils.get_datapath('figures')

import streamlit as st

st.title('Introduction')
st.write('Song popularity can be influenced by many factors, but one factor has garnered less interest \
         for consideration when predicting song popularity. This factor is the lyrical content of the \
         song. This lack of interest could stem from the extra work required for processing text data \
         and the fact that different genres can have differing vocabularies. Despite these difficulties, \
         this project aimed to see if it was possible to predict a song\'s popularity solely using the \
         lyrics alone.')

st.write('Limiting ourselves to only lyrics will make this objective harder to achieve, \
         but being able to meet this objective would bring value to the songwriting industry. \
         Specifically, our findings could enable individuals to write popular music more efficiently.')

st.write('You can look at the source code for this project at:')
st.write('https://github.com/0-Justin-ng/Lyrics_and_popularity')