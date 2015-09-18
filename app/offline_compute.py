from config import TOPICS, GLOVE_NAMES, GLOVE_VECS
from utils import read_csv, lemmatize_an_idea
import pickle
import random, json
from gloveSim import gloveSim
from itertools import combinations
import numpy as np
from scipy.spatial.distance import pdist, squareform

from gloveSim import VectorSpace

vector_space = VectorSpace.from_glovedata(300, GLOVE_NAMES, GLOVE_VECS)
NEAR_THRESHOLD = 0.5

THEME = 'weddingTheme'
PROP = 'weddingProp'
theme_out_name = 'theme_dict_set.p'
prop_out_name = 'prop_dict_set.p'
topics = {}
for k,v in TOPICS.iteritems():
    topics[k] = [(w[0], w[1].lower()) for w in read_csv(v)]

def get_dict(topic, outfile):
    X = np.array([vector_space.vec_for_sentence(w) for id,w in topics[topic] if not np.isnan(np.sum(vector_space.vec_for_sentence(w)))])
    glove_words = [t for t in topics[topic] if not np.isnan(np.sum(vector_space.vec_for_sentence(t[1])))]
    # pairs = list(combinations(range(len(glove_words)), 2))
    sim_matrix = 1 - squareform(pdist(X, 'cosine'))

    set_dict = {}
    for i in range(len(glove_words)):
        _t = []
        #find top 15
        idx_sim = [(j, sim_matrix[i][j]) for j in range(len(glove_words)) if j != i and sim_matrix[i][j]<NEAR_THRESHOLD]
        idx_sim = sorted(idx_sim, key=lambda t: t[1])
        to_consider = idx_sim[-15:]
        pairs = list(combinations(to_consider,2))
        random.shuffle(pairs)
        for j, k in pairs:
            assert(i!=j)
            assert(i!=k)
            if sim_matrix[j[0]][k[0]] < NEAR_THRESHOLD:
                _t.append((i,j[0],k[0]))
        set_dict[i] = _t
        if len(_t) < 3:
            print glove_words[i]
            # for a in to_consider:
            #     print glove_words[a[0]][1], a[1]


    to_save = {
        'set_dict': set_dict,
        'words': glove_words
    }
    # print to_save

    with open(outfile, 'w') as f:
        pickle.dump(to_save, f)




# get_dict(THEME, theme_out_name)
get_dict(PROP, prop_out_name)

