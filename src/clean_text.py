import pandas as pd
import numpy as np
import nltk
import string
import re

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

def get_pos(word):

    tag_map = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    tag = nltk.pos_tag([word])[0][1][0].upper()

    return tag_map.get(tag, wordnet.NOUN)

def lemmatize(text):
    return ' '.join(lemmatizer.lemmatize(
        word, get_pos(word)) for word in 
        nltk.word_tokenize(text))

def clean_text(text):
    #remove digits
    text = re.sub(r"\d+", "", text)
    #split words on hyphen
    text = re.sub(r"[.]?-[.]?", " ", text)
    #get rid of new lines
    text = text.replace("\n", " ")
    #remove punctuation
    translate_table = dict((ord(char), None) for char in string.punctuation) 
    text = text.translate(translate_table)
    #lowercase everything
    text = text.lower()
    #get rid of any other unwanted elements
    text = text.replace("\u2005", " ")
    #lemmatize
    text = lemmatize(text)

