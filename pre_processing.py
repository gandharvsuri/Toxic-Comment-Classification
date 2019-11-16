import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# TEXT ENRICHMENT REQUIRED OR NOT?

# Global counter for the number of comments
i = 0

''' Basic Cleaning of text. '''
def textPreProcessing(comment):
    # Convert the comments to lowercase
    comment = comment.lower()

    # Remove html markup
    comment = re.sub("(<.*?>)","",comment)

    # Remove non-ASCII and digits
    comment = re.sub("(\\W|\\d)"," ",comment)

    # Remove whitespace 
    comment = comment.strip()
    print("Text pre-processing...")
    return comment

''' Removing Stop Words -
    Words like "a", "an", "the", "on", "is", "all" etc. '''
def removeStopWords(comment):
    stop_words = set(stopwords.words('english'))
    clean_comment = ""
    for word in comment.split():
        if word not in stop_words:
            clean_comment += word
            clean_comment += " "
    print("Removing stop-words...")
    return clean_comment

''' Lemmatization - Reduce inflectional forms of words to a common base form.
    For example : books - book, looked - look. '''
# init lemmatizer
lemmatizer = WordNetLemmatizer()
def lemmatization(comment):
    lemmatizedComment = ""
    for word in comment.split():
        lemma = lemmatizer.lemmatize(word)
        lemmatizedComment += lemma
        lemmatizedComment += " "
    lemmatizedComment = lemmatizedComment.strip()
    print("Lemmatization...")
    return lemmatizedComment

''' Stemming - Reducing words to their word stem, base or root.
    Lemmatization uses lexical knowledge to get correct base forms of words.
    Stemming simply chops off infelctions. '''
# init stemmer
stemmer = SnowballStemmer("english")
def stemming(comment):
    stemmedComment = ""
    for word in comment.split():
        stem = stemmer.stem(word)
        stemmedComment += stem
        stemmedComment += " "
    stemmedComment = stemmedComment.strip()
    print("Stemming...")
    return stemmedComment

''' Identifying clean comments. '''
#def cleanComments(row):
    


''' Apply all the pre-processing functions. '''
def applyPreProcessing(data):
    comment = data["comment_text"]
    comment = textPreProcessing(comment)
    comment = removeStopWords(comment)
    comment = lemmatization(comment)
    comment = stemming(comment)

    # View the number of comments getting preprocessed
    global i
    i = i + 1
    print("Comments pre-processed : " + str(i) + " / 159571")    
    return comment


if __name__ == "__main__":
    train_data = pd.read_csv("Data/train.csv")
    test_data = pd.read_csv("Data/test.csv")
    print(train_data.info())
    train_data["comment_text"] = train_data.apply(lambda data : applyPreProcessing(data),axis = 1)
    test_data["comment_text"] = test_data.apply(lambda data : applyPreProcessing(data), axis = 1)
    # Identifying clean comments
    train_data["clean"] = [1 if x == 0 else 0 for x in np.sum(train_data.values == 1, 1)]
    print(train_data)
    print(test_data)
    train_data.to_csv(r'Data/ppc_train.csv')
    test_data.to_csv(r'Data/ppc_test.csv')
    # print(test_data)



