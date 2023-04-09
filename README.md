# Lyrics and Spotify Popularity

## Introduction
Song popularity can be influenced by many factors, but one factor has garnered less interest for consideration when predicting song popularity. This factor is the lyrical content of the song. This lack of interest could stem from the extra work required for processing text data and the fact that different genres can have differing vocabularies. Despite these difficulties, this project aimed to see if it was possible to predict a song’s popularity solely using the lyrics alone. 

Limiting ourselves to only lyrics will make this objective harder to achieve, but being able to meet this objective would bring value to the songwriting industry. Specifically, our findings could enable individuals to write popular music more efficiently.


## Data Source
The dataset can be found [here](https://www.cs.cornell.edu/~arb/data/genius-expertise/). Additional popularity data and track genre was obtained using the Spotify Web Developer API. The fully cleaned dataset (1.1 GB) used in this project can be found [here](https://drive.google.com/file/d/1JD0Oa3Iiv1husTWHd39OSpAicz12yl_C/view?usp=share_link).

Specifically this cleaned dataset contains the following features:
|Feature|Description|
|------|----|
|`song`| Genius.com URL identifier.|
`lyrics` | Raw lyrics.|
`release_year` | Release year of the song. |
`title` | Title of the song. |
`primary_artist` | Main artist associated with the song. |
`views` | Genius.com page views for a song. |
`cleaned_lyrics` | Lowercase lyrics with punctuation, special characters and tags removed. |
`language` | Song language. All songs are in English. This was used to help select for english songs. Language was determined using a `fasttext` pre-trained model. |
`log_scaled_views` | Log-transformed of Genius.com page views.|
`popular` | Binary output of popularity. If song had more than the median amount of page views on Genius.com it would be assigned 1 (popular), if not assign 0 (unpopular). Feature was used as target for preliminary modeling, but was not used for final modeling. 
`popularity_three_class` | Divided popularity into three classes using the 33rd and 66th percentile of Genius.com page views. These represent high (2), medium (1) and low (0) popularity. Feature was used as target for preliminary modeling, but was not used for final modeling. 
`cleaned_lyrics_stem` | Cleaned lyrics that were stemmed and had English stop words removed. These stop words were defined in `nltk.corpus.stopwords.words('english')`. |
`spotify_popularity` | A popularity rating assigned by Spotify that ranges from 0 to 100. This number is algorithmically determined by Spotify. Exact details are unknown, but it is claimed that the total plays and recency of plays of a song increase the popularity rating. 
`ada_embeddings` | Text embeddings that represent each lyric generated using the second generation OpenAI Ada text embedding model. The embeddings are stored as a string representation of a vector in the dataset. To convert them to an array you can use the function `utilities.utils.get_ada_embeddings()` found in the repository and run `df['ada_embeddings'].apply(get_ada_embeddings)` after loading the dataframe. 
`spotify_popuarity_three_class` | Divides `spotify_popularity` into thirds based on the 33rd and 66th percentile of `spotify_popularity`. This results in three classes: high (2), medium (1) and low (0) popularity. This target was used in the final modeling. 

## Findings
You can checkout the following two files in the repository for a summary of the project: 
- `Capstone_Final_Presentation.pdf` 
- `Capstone_Report.pdf`

## Test the Model
Do you think you can write a hit song? You can take a look at the model in action [here](https://0-justin-ng-lyrics-and-popularity-appintroduction-mcg5kw.streamlit.app/Predicting_Popularity). 

---

## Supplementary: Summary of the Contents of This Repository
This section will provide a summary of all the directories and files in this repository. 

`.streamlit` 
- `config.toml` - Config file for the Streamlit app theme.

`app`
- Files for the Streamlit app itself.
    - `app/pages`: Source code for the two pages in the Streamlit app.
        - `1_Predicting_Popularity.py` - Lets user input lyrics and predict a popularity class. 
        - `2_Exploring_Lyrics.py` - Provides more in-depth info in the dataset.
    - `Introduction.py`: Intro page for the app.
    - `requirements.txt` - Provides the dependencies for Streamlit to build the app on Streamlit Community Cloud.


`data`
- Directory for the data files. You can put the `clean_lyrics_final.csv` into this directory if you want to run some of the code in the notebooks. 
    - `data/language_detect/lid.176.bin` - Model used for detecting language. 


`figures`
- Contains plots used for the Streamlit app. 

`lexVec_data`
- You can unzip the following [file](https://drive.google.com/file/d/1NtOjkNtbevgg5xWkcop62NYY332tmiJh/view?usp=sharing) to get the LexVec embeddings for seperate words. This was used to generate one of the representations of the lyrics, by averaging the word embeddings. 

`model`
- Directory for all the `GridSearchCV` results. You can access the best model from the search using `GridSearchCV.best_estimator_`. All these models are trying to predict `spotify_popularity_three_class`.
    - `log_reg_ada_pca.pkl` - Logsitic regression with PCA of the Ada embeddings as input. 
    - `log_reg_ada.pkl` - Logsitic regression with the Ada embeddings as input. 
    - `log_reg_tfidf_hip_hop.pkl` - Logsitic regression with TF-IDF vectorizing only lyrics from hip hop songs as input. 
    - `log_reg_tfidf_nmf.pkl` - Logsitic regression with NMF of the TF-IDF vectorized lyrics as input. 
    - `log_reg_tfidf.pkl` - Logisitic regression with TF-IDF vectorized lyrics. This model was used for the Streamlit app. 
    - `mnb_tfidf_hip_hop.pkl` - Multinomial Naive Bayes classifier with TF-IDF vectorizing only lyrics from hip hop songs as input.
    - `naive_bayes_tf_idf.pkl` - Multinomial Naive Bayes with TF-IDF vectorized lyrics as input. 
    - `random_forest_tf_idf_v1.pkl` - Random forest with TF-IDF vectorized lyrics. 

`notebooks`
- This directory contains all the jupyter notebooks used for data collection, cleaning, analysis and model building and evaluation. 
    - `0_cleaning_lyrics.ipynb` - Outlines the steps used for cleaning the lyrics and converting the raw data into a csv. 
    - `1_eda.ipynb` - Provides general information in the dataset. Looks at the release year, genre, word distrbution, word count and some dimensionality reduction. 
    - `2_transforming_lyrics.ipynb` - Outlines the steps for vectorizing lyrics for preliminary modeling. The vectorizers involved are CountVectorizers and TF-IDF. Also created average document embeddings from LexVec word embeddings.
    - `3_prelim_modelling.ipynb` - Preliminary quick modelling to see which representations of the lyrics showed promise to move forward with. 
    - `4_generating_OpenAI_embeddings.ipynb` - Provides the steps for embedding the lyrics using the Ada text embedding model by OpenAI. 
    - `5_extracting_spotify_info.ipynb` - Provides steps for extracting track info using the Spotify API. 
    - `6_model_optimization` - Create pipelines to run hyperparameter optimization for various models. Utilized TF-IDF as the vectorizer in these pipelines. 
    - `7_model_evaluation` - Evaluated models based on various metrics, such as accuracy, precision, recall and AUC-ROC.

`spotify_scrape`
- Contains info from the Spotify scrape. You can load these using `joblib.load()`. 
    - `artist_genre_mapping.pkl` - Spotify only provides genres for artists and not individual tracks. Needed to map genres to an artist. 
    - `spotify_popularity_2.pkl` - Spotify popularity. Rate limit reached resulted in an error so needed to split the scrape into two parts. This is the second part. 
    - `spotify_popularity.pkl` - Spotify popularity. This is the first part of the scrape before the error. 

`utilities` 
- Package containing general functionality scripts used throughout the project. 
    - `__init__.py` - Initialization file for establishing a package. 
    - `utils.py` - Contains general scripts used for various aspects in the project. 
    - `vectorizer_pipeline.py` - Contains a class that stores and saves vectorizer output and the vectorizer itself. Additionally, allows for data splitting into train, validation and test sets. These splits are also stored. 

`vectorizer_data`
- Contains vectorizer data generated by `utilities.vectorizer_pipeline.py`. 

`Capstone_Final_Presentation.pdf` 
- A summary presentation on this project.

`Capstone_Report.pdf`
- A summary report on this project. 

`requirements.txt`
- Contains the minimal packages required to run the project. 
    


