from utils import cossim
import spacy.en
nlp = spacy.en.English()

def spacySim(word1, word2):
    tok1 = nlp(word1)[0]
    tok2 = nlp(word2)[0]
    sim = cossim(tok1.repvec, tok2.repvec)
    return float(sim)