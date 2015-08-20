from utils import cossim, lemmatize_an_idea
import spacy.en
import numpy as np
nlp = spacy.en.English()

def spacySim(word1, word2):
    tok1 = nlp(unicode(word1))[0]
    tok2 = nlp(unicode(word2))[0]
    sim = cossim(tok1.repvec, tok2.repvec)
    return float(sim)

def spacyPhraseSim(p1, p2):
    # TODO: find a more reasonable way to aggregate vector
    processed1 = ' '.join(lemmatize_an_idea(p1))
    processed2 = ' '.join(lemmatize_an_idea(p2))
    tok1 = nlp(unicode(processed1))

    tok2 = nlp(unicode(processed2))

    v1 = np.mean([t.repvec for t in tok1], axis=0)
    v2 = np.mean([t.repvec for t in tok2], axis=0)
    sim = cossim(v1, v2)
    return float(sim)
