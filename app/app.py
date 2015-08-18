from flask import Flask, jsonify
from utils import read_csv
from config import TOPICS
# from spacySim import spacySim
from gloveSim import gloveSim

app = Flask(__name__)

# --- Init vocab (TODO: really hacky, fix this later)
# --- Assume 1 word per line csv
topics = {}
for k,v in TOPICS.iteritems():
    topics[k] = list(set([w[0].lower() for w in read_csv(v)]))
@app.route('/spaCy/similarity/<word1>/<word2>', methods=['GET'])
def get_spacy_sim(word1, word2):
    return jsonify(
            word1 = word1,
            word2 = word2,
            similarity = spacySim(word1, word2))

@app.route('/GloVe/similarity/<word1>/<word2>', methods=['GET'])
def get_glove_sim(word1, word2):
    return jsonify(
            word1 = word1,
            word2 = word2,
            similarity = gloveSim(word1, word2))

def get_top15(word, vocab_list, func):
    sim_vec = [(w, func(word,w)) for w in vocab_list]
    sim_vec = sorted(sim_vec, key=lambda t: t[1])
    return jsonify(
            word = word,
            similar = [i for i in reversed(sim_vec[-15:])],
            different = sim_vec[:15])

@app.route('/spaCy/top15/<topic>/<word>', methods=['GET'])
def get_spacy_top15(topic, word):
    return get_top15(word, topics[topic], spacySim)

@app.route('/GloVe/top15/<topic>/<word>', methods=['GET'])
def get_glove_top15(topic, word):
    return get_top15(word, topics[topic], gloveSim)


if __name__ == '__main__':
    app.run(debug=True)
