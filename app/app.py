from flask import Flask, jsonify
from spacySim import spacySim
from gloveSim import gloveSim

app = Flask(__name__)

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

@app.route('/spaCy/top15/<topic>/<word>', methods=['GET'])
def get_spacy_top15(topic, word):
    return jsonify(
            word = word,
            similar = [],
            different = [])

@app.route('/GloVe/top15/<topic>/<word>', methods=['GET'])
def get_glove_top15(topic, word):
    return jsonify(
            word = word,
            similar = [],
            different = [])


if __name__ == '__main__':
    app.run(debug=True)


# jsonify(items=[dict(a=1, b=2), dict(c=3, d=4)])
# {"items": [{"a": 1, "b": 2}, {"c": 3, "d": 4}]}
