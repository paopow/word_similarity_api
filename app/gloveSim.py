from utils import cossim
import numpy as np
from gensim.utils import lemmatize

#TODO: put this in config file
GLOVE_NAMES = '/Users/paopow/Data/iis-dev/Data/GloVe/6B.300d.names'
GLOVE_VECS = '/Users/paopow/Data/iis-dev/Data/GloVe/6B.300d.npy'

def lemmatize_an_idea(idea, use_stoplist=True):
    if use_stoplist:
        lemm = [lem[:-3] for lem in lemmatize(idea) if lem[:-3] not in stoplist]
    else:
        lemm = [lem[:-3] for lem in lemmatize(idea) if lem[:-3]]
    return lemm

def lemmatize_ideas(ideas):
    lemms = [lemmatize_an_idea(raw) for raw in ideas]
    return lemms


class VectorSpace:
    def __init__(self, names, vecs):
        self.names = names
        self.vecs = vecs

    def __getitem__(self, word):
        return self.vecs[self.names.get_loc(word)]

    def __contains__(self, word):
        return word in self.names

    def vec_for_tokens(self, tokens):
        return np.mean([self[token] for token in tokens if token in self], axis=0)

    def vec_for_sentence(self, sentence):
        tokens = lemmatize_an_idea(sentence,False)
        return self.vec_for_tokens(tokens)

    @classmethod
    def from_glovedata(cls, num_dims, names_filename, data_filename):
        import pandas as pd
        names = pd.Index(line.strip() for line in open(names_filename))
        num_terms = len(names)
        vecs = np.memmap(data_filename, dtype=np.float32, mode='r', shape=(num_terms, num_dims))
        return cls(names, vecs)

vector_space = VectorSpace.from_glovedata(300, GLOVE_NAMES, GLOVE_VECS)

def gloveSim(tokens1, tokens2):
    vec1 = vector_space.vec_for_sentence(tokens1)
    vec2 = vector_space.vec_for_sentence(tokens2)
    return float(cossim(vec1,vec2))
