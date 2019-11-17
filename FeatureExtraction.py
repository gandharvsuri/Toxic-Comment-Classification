import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('./Data/ppc_train.csv')
test_data = pd.read_csv('./Data/ppc_test.csv')

train_text = data['comment_text']
test_text = test_data['comment_text']
all_text = pd.concat([train_text, test_text])

def word_vec():
    global all_text
    global train_text
    global test_text
    word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 2),min_df=2,max_df=0.5,
    max_features=60000
    )

    word_vectorizer.fit(all_text)
    train_word_features = word_vectorizer.transform(train_text)
    test_word_features = word_vectorizer.transform(test_text)

    return(train_word_features,test_word_features)

def char_vec():
    global all_text
    global train_text
    global test_text
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