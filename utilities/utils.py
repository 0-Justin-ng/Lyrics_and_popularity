import re
import json
import os
from pathlib import Path
import joblib

import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk import download

# Set the english stop words.
download('stopwords')
ENGLISH_STOP_WORDS = set(stopwords.words('english'))

# Set the directory name for the vectorizer data. 
VECTORIZER_PATH = 'vectorizer_data'


def get_datapath(source_directory):
    '''
    Returns the absolute path to a source directory. This allows for filepaths
    to not be hard coded in. 

    Parameters
    ----------
    source_directory : str
        The directory that is to be accessed. 

    Returns
    -------
    absolute_path : Path Object
        This is the absolute path to the directory of interest. 

    Examples
    --------
    >>> from utilities.utils import get_datapath
    >>> DATA_PATH = get_datapath('data')
    >>> DATA_PATH
        PosixPath('~/repository/path/on/local/system/data')

    '''
    cwd = os.getcwd()
    parent_path = Path(cwd).parent

    absolute_path = parent_path / source_directory

    return absolute_path


def json_loader(file_path):
    '''
    Takes a file path and loads a json line by line.
    Codes was adapted from here:
    https://github.com/cptq/genius-expertise/blob/master/load_data.py

    Parameters
    ----------
    file_path : str
        The directory that houses the json file. 

    Returns
    -------
    pd.DataFrame
        A flattened version of the json data. 

    '''
    
    # Reads the json file line by line.
    with open(file_path) as f:
        lines = f.read().splitlines()

    # Create an intermediate dataframe to store the lines.  
    df_inter = pd.DataFrame(lines)
    # Add a column for the json elements. 
    df_inter.columns = ['json_element']
    # Load the json for each entry into the new column
    df_inter['json_element'].apply(json.loads)

    # Returns a flattened version of the json information.
    return pd.json_normalize(df_inter['json_element'].apply(json.loads))


def load_transformed_data(vectorizer_name):
    '''
    Loads the data generated from vectorizer_pipeline.VectorizerPipeline().

    Parameters
    ----------
    vectorizer_name : str
        The name of the vectorizer that you want to obtain the data from. 

    Returns
    -------
    (array-like)
        The train, validation and test splits generated from VectorizerPipeline().
         
    '''
    with open(
        get_datapath(VECTORIZER_PATH) / vectorizer_name / 'data.pkl', 'rb'
        ) as f:
        data = joblib.load(f)

    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_vectorizer(vectorizer_name):
    '''
    Loads the vectorizer generated from vectorizer_pipeline.VectorizerPipeline().

    Parameters
    ----------
    vectorizer_name : str
        The name of the vectorizer that you want to obtain the vectorizer from. 

    Returns
    -------
    vectorizer : sklearn.feature_extraction.text Vectorizer Object
        The vectorizer generated used in the VectorizerPipeline().
         
    '''
    with open(
        get_datapath(VECTORIZER_PATH) / vectorizer_name / f'{vectorizer_name}.pkl', 'rb'
        ) as f:
        vectorizer = joblib.load(f)

    return vectorizer


def clean_lyrics(lyric):
    '''
    This function takes a str that represents each lyric and outputs a cleaned version of the lyric.

    Parameters
    ----------
    lyric : array-like of str

    Returns
    -------
    cleaned_lyric : str
        The cleaned version of the lyric with punctuation, tags and excess whitespace removed. 

    Examples
    -------
    >>> from utilities.utils import clean_lyrics
    >>> test_string = '[Chorus] I like to look at lyrics   ... Sometimes.'
    >>> clean_string = clean_lyrics(test_string)
    >>> clean_string
    'i like to look at lyrics sometimes' 
        
    '''
    # Lowercase lyrics and replace '\n' and apostrophes. 
    cleaned_lyric = lyric.lower().replace('\n', ' ').replace("\'",'')
    
    # Deal with tags.
    cleaned_lyric = re.sub(
        # (\[\w+\]) Matches any occurrence of [any word characters] or 
            # This matches all tags that are not the [Verse #] tag
        # (\[\w+ \d+\]) Matches any occurrence of [any word characters and any digit]
            # This is to match the [Verse #] tag
        '(\[(.*?)\])',
        '',
        cleaned_lyric
    )

    # Remove all punctuation.
    cleaned_lyric = re.sub(
        "[;:!><\"\',/@#$%&?–*+()|]",
        '',
        cleaned_lyric
    )
    
    # Captures other punctuation that can not be replaced with the above regex.
    other_punctuation = ['-', '—','...', '”', '“', '.', '…', '’', '^', 'ι', '‘']
    for punctuation in other_punctuation:
        if punctuation == '—': 
            # This is to add spaces to large hyphens which denote breaks in the lyrics
            # not hyphenated words, so we want to separate these words.
            cleaned_lyric = cleaned_lyric.replace(punctuation, ' ')
        else:
            cleaned_lyric = cleaned_lyric.replace(punctuation, '')
  
    # Remove whitespaces greater than one. 
    cleaned_lyric = re.sub(
        ' {2,}', ' ',
        cleaned_lyric
    )

    return cleaned_lyric


def stop_word_removal_and_stem(lyrics):
    '''
    Removes english stop words as defined by nltk and also stems the lyrics.

    Parameters
    ----------
    lyrics : str
        The lyrics to be modified. 

    Returns
    -------
    str
        The modified lyrics.
    
    Examples
    --------   
    >>> from utilities.utils import stop_word_removal_and_stem
    >>> test_string = 'a fox is running'
    >>> changed_string = stop_word_removal_and_stem(test_string)
    >>> changed_string
    'fox run' 
    '''

    # Split the lyrics into a list where each index holds a word.
    tokenized_lyrics = lyrics.split(' ')

    # Try a SnowballStemmer that is less aggressive than the Porter stemmer. 
    stemmer = SnowballStemmer('english')

    # Attempt to stem all words that are not part of the ENGLISH_STOP_WORDS.
    lyrics_modified = [
        stemmer.stem(word) 
        for word in tokenized_lyrics
        if word not in ENGLISH_STOP_WORDS
    ]

    # Return a string with the modfied lyrics.
    return ' '.join(lyrics_modified)


def _convert_ada_embeddings(embedding):
    '''
    This function converts an Ada embedding from a string
    to a numpy array.

    Parameters
    ----------
    embedding : str
        A single Ada embeding stored as a string, where each index is a string
        in the format '[embedding_dimension_1, embedding_dimension_2 ...]' 
    
    Returns
    -------
    np.array
        Converted embedding vector from a string to an np.array. 

    '''
    # Remove the brackets and commas in the string. 
    ada_embedding = embedding.replace('[', '').replace(']', '').replace(',', '')

    # Turn the embeddings into a list, where each element is one dimension of the embedding. 
    converted_ada_embedding = ada_embedding.split(' ')
    # Convert each value in the embedding from a str to a float. 
    converted_ada_embedding = [
        float(weight)
        for weight in converted_ada_embedding 
    ]
    
    return np.array(converted_ada_embedding)


def get_ada_embeddings(embeddings):
    '''
    This function stacks all the converted np.arrays. 

    Parameters
    ----------
    embeddings : array-like of str
        The Ada embeddings stored as an array of strings. 
    
    Returns
    -------
    np.array
        Stack of all converted ada embeddings.  

    '''
    
    # Convert the Ada embeddings into a numpy.array.
    ada_embeddings = [
        _convert_ada_embeddings(embedding)
        for embedding in embeddings
    ]
    # Stack the series and return the stacked array for modeling. 
    return np.stack(ada_embeddings, axis=0)


def load_model(model_path):
    '''
    '''
    with open(model_path, 'rb') as file:
        model = joblib.load(file)
    
    return model


def generate_multi_class_roc(y_score, y_onehot_test, n_classes, model_name):
    '''
    Generates a multi class ROC curve plot. 
    This code was adapted from: 
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    Parameters
    ----------
    y_score : array-like of shape (n samples in test set, n classes)
        The predicted probabilities for each class for each sample in the test set. 
    
    y_onehot_test : array-like of shape (n samples in test set, n classes)
        One-hot encoded output of the true class in the test set.

    n_classes : int
        The number of classes in the target.

    model_name : str
        Used for labeling the plots. 

    Returns
    -------
    None
        There is no return. this function just plots the multi class ROC curve.     

    '''

   
    fpr = dict() # Stores the false positive rates for each class.
    tpr = dict() # Stores the true positve rates for each class.
    roc_auc = dict() # Stores the auc for each class. 

    # Generate the ROC curve data using the roc_curve function from sklearn.
    # Do this for each class. 
    for i in range(n_classes):
        # Get the false positive and true positive rate, by looking at the column in the 
        # one-hot encoded true value for the desired class and the probability that the model predicts
        # for that desired class.
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:,i], y_score[:, i])
        # Compute the auc for the ROC curve for each class.
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area. The micro-average is the weighted average based
    # on the prevalence of a class. 
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    # Plot ROC curve
    plt.figure(figsize=(8,8))
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                    ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Reciever Operating Characteristc Curve for \n{model_name}')
    plt.legend(loc="lower right")
    plt.show()