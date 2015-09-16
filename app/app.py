from flask import Flask, jsonify, request
from utils import read_csv, lemmatize_an_idea
import json, random
from config import TOPICS
# from spacySim import spacySim, spacyPhraseSim
from gloveSim import gloveSim


app = Flask(__name__)

# --- Init vocab (TODO: really hacky, fix this later)
# --- Assume 1 word per line csv
topics = {}
for k,v in TOPICS.iteritems():
    topics[k] = [(w[0], w[1].lower()) for w in read_csv(v)]

TOPIC_STAT = {
    'weddingTheme': {
        'sim_mean': 0.111250152359,
        'sim_sd': 0.123640928544
    },
    'weddingProp': {
        'sim_mean': 0.141468741736,
        'sim_sd': 0.129488421166
    }

}

# @app.route('/spaCy/similarity', methods=['GET', 'POST'])
# def get_spacy_sim():
#     data = request.get_json()
#     words = data['words']
#     return jsonify(
#             words = words,
#             similarity = spacyPhraseSim(words[0]['text'], words[1]['text']))


@app.route('/GloVe/similarity', methods=['GET', 'POST'])
def get_glove_sim():
    data = request.get_json()
    words = data['words']
    return jsonify(
            words = words,
            similarity = gloveSim(words[0]['text'], words[1]['text']))


def get_top15(word, vocab_list, func):
    sim_vec = [{'id':w[0], 'text': w[1], 'similarity': func(word,w[1])}
        for w in vocab_list if ' '.join(lemmatize_an_idea(w[1])) != ' '.join(lemmatize_an_idea(word)) and func(word,w[1]) > -100]
    sim_vec = sorted(sim_vec, key=lambda t: t['similarity'])
    return jsonify(
            word = word,
            similar = [i for i in reversed(sim_vec[-15:])],
            different = sim_vec[:15])


# @app.route('/spaCy/top15/<topic>', methods=['GET', 'POST'])
# def get_spacy_top15(topic):
#     data = request.get_json()
#     word = data['word']
#     return get_top15(word['text'], topics[topic], spacyPhraseSim)


@app.route('/GloVe/top15/<topic>', methods=['GET', 'POST'])
def get_glove_top15(topic):
    data = request.get_json()
    word = data['word']
    return get_top15(word['text'], topics[topic], gloveSim)


@app.route('/GloVe/simSet/<topic>', methods=['GET', 'POST'])
def get_glove_sim_set(topic):
    data = request.get_json()
    word = data['word']
    return get_sim_set(word['text'], topics[topic], topic, gloveSim)


def is_diverse_in_range(triple, topic, func):
    w1 = triple[0]['text']
    w2 = triple[1]['text']
    w3 = triple[2]['text']
    lower_bound = TOPIC_STAT[topic]['sim_mean'] - TOPIC_STAT[topic]['sim_sd']
    upper_bound = TOPIC_STAT[topic]['sim_mean'] + TOPIC_STAT[topic]['sim_sd']
    w12_sim = func(w1,w2)
    w12_in_range = (w12_sim >= lower_bound and w12_sim <= upper_bound)
    w13_sim = func(w1,w3)
    w13_in_range = (w13_sim >= lower_bound and w13_sim <= upper_bound)
    w23_sim = func(w2,w3)
    w23_in_range = (w23_sim >= lower_bound and w23_sim <= upper_bound)
    return (w12_in_range and w23_in_range and w13_in_range)

from itertools import combinations
def get_sim_set(word, vocab_list, topic, func):
    sim_vec = [{'id':w[0], 'text': w[1], 'similarity': func(word,w[1])}
        for w in vocab_list if ' '.join(lemmatize_an_idea(w[1])) != ' '.join(lemmatize_an_idea(word)) and func(word,w[1]) > -100]
    sim_vec = sorted(sim_vec, key=lambda t: t['similarity'])
    most_similar = [i for i in reversed(sim_vec[-15:])]
    most_different = sim_vec[:15]
    max_itr = 150
    sim_sets = []
    count = 0
    for t in combinations(most_similar,3):
        if is_diverse_in_range(t, topic, func):
            sim_sets.append(t)
            if len(sim_sets) >= 5:
                break
        if count > max_itr:
            break
        count += 1

    diff_sets = []
    count = 0
    for t in combinations(most_similar,3):
        if is_diverse_in_range(t, topic, func):
            diff_sets.append(t)
            if len(diff_sets) >= 5:
                break
        if count > max_itr:
            break
        count += 1

    return jsonify(
            word = word,
            similar = most_similar,
            similar_set = sim_sets,
            different = most_different,
            different_set = diff_sets)




if __name__ == '__main__':
    app.run(debug=True)
