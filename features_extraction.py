import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

data = pd.read_csv('./Data/ppc_train.csv')
test_data = pd.read_csv('./Data/ppc_test.csv')

def word_vec(train_text,test_text):
    word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 2),min_df=2,max_df=0.5,
    max_features=60000
    )

    word_vectorizer.fit(train_text)
    word_vectorizer.fit(test_text)
    train_word_features = word_vectorizer.transform(train_text)
    test_word_features = word_vectorizer.transform(test_text)

    return(train_word_features,test_word_features)

def char_vec(train_text,test_text):
    char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    token_pattern=None,
    min_df=5,
    ngram_range=(2, 4),
    max_features=23000)


    char_vectorizer.fit(train_text)
    char_vectorizer.fit(test_text)
    train_char_features = char_vectorizer.transform(train_text)
    test_char_features = char_vectorizer.transform(test_text)

    return(train_char_features,test_char_features)



def get_features(train_text,test_text):
    
    train_word_features,test_word_features = word_vec(train_text,test_text,)
    train_char_features,test_char_features = char_vec(train_text,test_text)
    train_features = hstack([train_word_features, train_char_features]).tocsr()
    test_features = hstack([test_word_features, test_char_features]).tocsr()

    return (train_features,test_features)