from flask import Flask, jsonify
import spacy.en
from numpy import dot
from numpy.linalg import norm

app = Flask(__name__)
nlp = spacy.en.English()

def cossim(a, b):
    return dot(a, b) / (norm(a) * norm(b))

@app.route('/')
def index():
    return "Hello, World!"


@app.route('/spaCy/api/similarity/<word1>/<word2>', methods=['GET'])
def get_spacy_sim(word1, word2):
    tok1 = nlp(word1)[0]
    tok2 = nlp(word2)[0]
    sim = cossim(tok1.repvec, tok2.repvec)
    print type(sim)
    return jsonify({'word1': word1, 'word2': word2, 'similarity': float(sim)})


if __name__ == '__main__':
    app.run(debug=True)

