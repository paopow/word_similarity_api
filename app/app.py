from flask import Flask, jsonify

from spacySim import spacySim
app = Flask(__name__)


@app.route('/')
def index():
    return "Hello, World!"

@app.route('/spaCy/api/similarity/<word1>/<word2>', methods=['GET'])
def get_spacy_sim(word1, word2):
    return jsonify({
            'word1': word1,
            'word2': word2,
            'similarity': spacySim(word1, word2)})


if __name__ == '__main__':
    app.run(debug=True)

