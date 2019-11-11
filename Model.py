#!/usr/bin/env python
# coding: utf-8

# In[67]:


# Required library imports
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge
import pickle
from sklearn.model_selection import cross_val_score


from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression
from skmultilearn.problem_transform import LabelPowerset


# In[14]:


data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# In[15]:


#data.head()


# In[16]:


#test_data.head()


# In[17]:


#data.isnull().sum()


# In[18]:


#data.info()


# In[19]:


# function to identify comment_text which are clean. 
def clean_comments(row):
    if row['toxic'] == 1:
        return 0
    if row['severe_toxic'] == 1:
        return 0
    if row['obscene'] == 1:
        return 0
    if row['threat'] == 1:
        return 0
    if row['insult'] == 1:
        return 0
    if row['identity_hate'] == 1:
        return 0
    else:
        return 1


# In[20]:


# all those comment_text which don't lie in any of the categories
#data['Clean'] = data.apply(lambda row : clean_comments(row), axis = 1)


# In[21]:


#data.head()


# In[22]:


class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


# In[23]:



#labels = data[class_names]
#fig_size = plt.rcParams["figure.figsize"]
#fig_size[0] = 10
#fig_size[1] = 8
#plt.rcParams["figure.figsize"] = fig_size

#labels.sum(axis=0).plot.bar()


# In[24]:


#corr_matrix = data.corr()


# In[25]:


#sns.heatmap(corr_matrix, annot = True)


# In[26]:


# toxic & obscene
# toxic & insult
# insult & onscene


# In[27]:


# text length
def text_len(row):
    return len(row['comment_text'])


# In[28]:


#data['text_length'] = data.apply(lambda row : text_len(row), axis = 1)


# In[29]:


tokenizer = RegexpTokenizer(r'\w+')


# First process of pre processing to remmove unwanted html tags, lower the text, and remove punctuations which later create difficulty while creating vectors.

# In[30]:


def pre_process_text(sentence):
    
    # lower case
    sentence = sentence.lower()
    
    # Remove html-tags
    cleanr = re.compile('<.*?>')
    sentence = re.sub(cleanr, ' ', str(sentence))

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    sentence = re.sub(r'[.|,|)|(|\|/]',r' ',sentence)
    sentence = sentence.strip()
    sentence = sentence.replace("\n"," ")

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


# Remove stop words so that we form better vectors as stop words have a higher frequency and are not much informative.

# In[36]:


stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)

def remove_stopwords(sentence):
    global re_stop_words
    sentence = re_stop_words.sub(" ", sentence)
    
    return sentence


# stemming using SnowballStemmer

# In[37]:


stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence


# In[38]:


def pre_process_apply(row):
    sentence = row['comment_text']
    
    sentence = pre_process_text(sentence)
    sentence = remove_stopwords(sentence)
    sentnece = stemming(sentence)
    
    return sentence


# In[39]:


data['processed_comment_text'] = data.apply(lambda row : pre_process_apply(row), axis = 1)


# In[40]:


test_data['processed_comment_text'] = data.apply(lambda row : pre_process_apply(row) , axis = 1)


# In[41]:


train_text = data['processed_comment_text']
test_text = test_data['processed_comment_text']
all_text = pd.concat([train_text, test_text])


# In[43]:


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


# In[44]:


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


# In[58]:


#x_ train and x_test
train_features = hstack([train_word_features, train_char_features]).tocsr()
test_features = hstack([test_word_features, test_char_features]).tocsr()


# In[59]:


y_train = data.drop(labels = ['id','comment_text','Clean','text_length','processed_comment_text'], axis=1)
#y_train


# In[60]:


submission = pd.DataFrame.from_dict({'id': test_data['id']})


# In[68]:


# initialize label powerset multi-label classifier
classifier = LabelPowerset(LogisticRegression())


# In[69]:


# Training logistic regression model on train data
classifier.fit(train_features, y_train)


# In[70]:


# predict
predictions = classifier.predict(test_features)
# accuracy
predictions


# In[ ]:




