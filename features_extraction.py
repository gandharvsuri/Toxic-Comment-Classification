import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import regex as re
import regex, string

data = pd.read_csv('./Data/ppc_train.csv')
test_data = pd.read_csv('./Data/ppc_test.csv')

def word_vec(train_text,test_text,all_text):
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        tokenizer=lambda x: regex.findall(r'[^\p{P}\W]+', x),
        analyzer='word',
        token_pattern='(?u)\\b\\w\\w+\\b\\w{,1}',
        min_df=5,
        norm='l2',
        ngram_range=(1, 1),
        max_features=30000)

    word_vectorizer.fit(all_text)
    train_word_features = word_vectorizer.transform(train_text)
    test_word_features = word_vectorizer.transform(test_text)

    return(train_word_features,test_word_features)

def char_vec(train_text,test_text,all_text):
    char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    token_pattern=None,
    min_df=5,
    ngram_range=(2, 4),
    max_features=23000)

    char_vectorizer.fit(all_text)
    train_char_features = char_vectorizer.transform(train_text)
    test_char_features = char_vectorizer.transform(test_text)

    return(train_char_features,test_char_features)



def get_features(train_text,test_text,all_text):
    train_word_features,test_word_features = word_vec(train_text,test_text,all_text)
    train_char_features,test_char_features = char_vec(train_text,test_text,all_text)
    train_features = hstack([train_word_features, train_char_features]).tocsr()
    test_features = hstack([test_word_features, test_char_features]).tocsr()

    return (train_features,test_features)