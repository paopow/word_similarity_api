from flask import Flask, jsonify, request
from utils import read_csv
import json
from config import TOPICS
from spacySim import spacySim
from gloveSim import gloveSim

app = Flask(__name__)

# --- Init vocab (TODO: really hacky, fix this later)
# --- Assume 1 word per line csv
topics = {}
for k,v in TOPICS.iteritems():
    topics[k] = list(set([w[0].lower() for w in read_csv(v)]))


@app.route('/spaCy/similarity', methods=['GET', 'POST'])
def get_spacy_sim():
    data = request.get_json()
    words = data['words']
    return jsonify(
            words = words,
            similarity = spacySim(words[0]['text'], words[1]['text']))


@app.route('/GloVe/similarity', methods=['GET', 'POST'])
def get_glove_sim():
    data = request.get_json()
    words = data['words']
    return jsonify(
            words = words,
            similarity = gloveSim(words[0]['text'], words[1]['text']))


def get_top15(word, vocab_list, func):
    sim_vec = [(w, func(word,w)) for w in vocab_list]
    sim_vec = sorted(sim_vec, key=lambda t: t[1])
    return jsonify(
            word = word,
            similar = [i for i in reversed(sim_vec[-15:])],
            different = sim_vec[:15])


@app.route('/spaCy/top15/<topic>', methods=['GET', 'POST'])
def get_spacy_top15(topic):
    data = request.get_json()
    word = data['word']
    return get_top15(word['text'], topics[topic], spacySim)


@app.route('/GloVe/top15/<topic>', methods=['GET', 'POST'])
def get_glove_top15(topic):
    data = request.get_json()
    word = data['word']
    return get_top15(word['text'], topics[topic], gloveSim)


if __name__ == '__main__':
    app.run(debug=True)
