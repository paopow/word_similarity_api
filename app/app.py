from flask import Flask, jsonify, request
from utils import read_csv, lemmatize_an_idea
import json, random
from config import TOPICS
import pickle
# from spacySim import spacySim, spacyPhraseSim
from gloveSim import gloveSim

with open('theme_dict_set.p') as f:
    theme_dict_set = pickle.load(f)

with open('prop_dict_set.p') as f:
    prop_dict_set = pickle.load(f)

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
    for_sim = [t for t in sim_vec if t['similarity'] < 0.5]
    return jsonify(
            word = word,
            similar = [i for i in reversed(for_sim[-15:])],
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
    word = data['word']['text']
    func = gloveSim
    this_dict_set = theme_dict_set if topic=='weddingTheme' else prop_dict_set
    vocab_list = this_dict_set['words']
    sim_vec = [{'id':w[0], 'text': w[1], 'similarity': func(word,w[1])}
        for w in vocab_list if ' '.join(lemmatize_an_idea(w[1])) != ' '.join(lemmatize_an_idea(word)) and func(word,w[1]) > -100]

    sim_vec = sorted(sim_vec, key=lambda t: t['similarity'])
    for_sim = [t for t in sim_vec if t['similarity'] < 0.5]
    operation = data['operation']
    similar_sets = []
    different_sets = []

    if operation == 'similar':
        similar_words = [i for i in reversed(for_sim[-5:])]
        for s in similar_words:
            s_idx = vocab_list.index((s['id'], s['text']))
            tmp = random.choice(this_dict_set['set_dict'][s_idx])
            tmp = (
                {'id': vocab_list[tmp[0]][0], 'text': vocab_list[tmp[0]][1]},
                {'id': vocab_list[tmp[1]][0], 'text': vocab_list[tmp[1]][1]},
                {'id': vocab_list[tmp[2]][0], 'text': vocab_list[tmp[2]][1]},
                )
            similar_sets.append(tmp)
    else:
        different_words = sim_vec[:5]
        for s in different_words:
            s_idx = vocab_list.index((s['id'], s['text']))
            tmp = random.choice(this_dict_set['set_dict'][s_idx])
            tmp = (
                {'id': vocab_list[tmp[0]][0], 'text': vocab_list[tmp[0]][1]},
                {'id': vocab_list[tmp[1]][0], 'text': vocab_list[tmp[1]][1]},
                {'id': vocab_list[tmp[2]][0], 'text': vocab_list[tmp[2]][1]},
                )
            different_sets.append(tmp)

    return jsonify(
            word = data['word'],
            similar = similar_sets,
            different = different_sets)


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



def get_sorted_similar(word, vocab_list, func):
    sim_vec = [{'id':w[0], 'text': w[1], 'similarity': func(word,w[1])}
        for w in vocab_list if ' '.join(lemmatize_an_idea(w[1])) != ' '.join(lemmatize_an_idea(word)) and func(word,w[1]) > -100]
    sim_vec = sorted(sim_vec, key=lambda t: t['similarity'])
    for_sim = [t for t in sim_vec if t['similarity'] < 0.5]
    return for_sim


SET_SIZE = 1
def _get_sim_set(word, vocab_list, topic, func):
    sim_vec = get_sorted_similar(word, vocab_list, func)
    sim_candidates = sim_vec[-15:]

    max_itr = 100
    sim_set = None
    for i in range(max_itr):
        t1,t2 = random.sample(sim_candidates,2)
        w1 = t1['text']
        w2 = t2['text']
        if func(w1, w2) < 0.5:
            sim_set = [t1,t2]

    return sim_set




from itertools import combinations
def get_sim_set(word, vocab_list, topic, func):
    sim_vec = get_sorted_similar(word, vocab_list, func)
    sim_candidates = sim_vec[-15:]
    diff_candidates = sim_vec[:15]

    max_itr = 100
    sim_sets = []
    for i in range(max_itr):
        t1,t2,t3 = random.sample(sim_candidates,3)
        w1 = t1['text']
        w2 = t2['text']
        w3 = t3['text']
        if func(w1, w2) < 0.5 and func(w1,w3) < 0.5 and func(w2,w3) < 0.5:
            sim_sets.append([t1,t2,t3])

        if len(sim_sets) > SET_SIZE:
            break

    diff_sets = []
    diff_start_points = random.sample(diff_candidates, SET_SIZE + 5)
    for s in diff_start_points:
        buddies = _get_sim_set(s['text'],vocab_list, topic, func)
        if buddies is not None:
            diff_sets.append([s,buddies[0], buddies[1]])
        if len(diff_sets) > SET_SIZE:
            break
    return jsonify(
            word = word,
            similar_set = sim_sets,
            different_set = diff_sets)




if __name__ == '__main__':
    app.run(debug=True)
