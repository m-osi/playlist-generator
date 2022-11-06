from clean_text import clean_text
from nltk.corpus import wordnet
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import zipfile
import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')

with zipfile.ZipFile('.\data\glove\glove.6B.50d.zip', 'r') as zip_ref:
    zip_ref.extractall('.\data\glove')


def embedding_for_vocab(filepath, word_index,
                        embedding_dim):
    vocab_size = len(word_index) + 1

    # Adding again 1 because of reserved 0 index
    embedding_matrix_vocab = np.zeros((vocab_size,
                                       embedding_dim))

    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix_vocab[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix_vocab


class CleanText(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self.columns] = X[self.columns].apply(clean_text)
        return X


class Group(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X[self.columns]
        X = X.groupby(
            self.columns[0]).agg(
                list).reset_index()
        X[self.columns[1]] = X[self.columns[1]].apply(
            lambda x: ' '.join(x))
        X = X[self.columns[1]].tolist()
        return X


class Tokenize(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        all_lyrics = ' '.join(list(X[self.columns]))
        vocab = list(set(nltk.word_tokenize(all_lyrics)))
        word_index = {vocab[i-1]: i for i in range(1, len(vocab)+1)}
        return word_index


class Embed(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        embedding_matrix_vocab = embedding_for_vocab(
            '.\data\glove\glove.6B.50d.txt', X, 50)
        return embedding_matrix_vocab
