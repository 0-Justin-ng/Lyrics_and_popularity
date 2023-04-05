import pandas as pd
import numpy as np
import streamlit as st
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent.parent))

from utilities import utils
import joblib


import streamlit as st
st.title("Let's Explore Some Lyrics")
st.write('Here you will get a closer look into the lyrics of your favourite artists and songs.')