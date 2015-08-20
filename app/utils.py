from numpy import dot
from numpy.linalg import norm
from gensim.utils import lemmatize
from nltk.corpus import stopwords
import csv
stoplist = stopwords.words('english')

def cossim(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def read_csv(filename, has_header = True):
    items = []
    with open(filename, 'rU') as f:
        reader = csv.reader(f)
        if has_header:
            reader.next()

        for row in reader:
            items.append(row)
    return items

def lemmatize_an_idea(idea, use_stoplist=True):
    if use_stoplist:
        lemm = [lem[:-3] for lem in lemmatize(idea) if lem[:-3] not in stoplist]
    else:
        lemm = [lem[:-3] for lem in lemmatize(idea) if lem[:-3]]
    return lemm

def lemmatize_ideas(ideas):
    lemms = [lemmatize_an_idea(raw) for raw in ideas]
    return lemms
