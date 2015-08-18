from flask import jso
from numpy import dot
from numpy.linalg import norm
import csv

def cossim(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def read_csv(filename, has_header = True):
    items = []
    with open(filename, 'rU') as f:
        reader = csv.reader(f)
        if has_header:
            reader.next()

        for row in reader:
            items.append(row)
    return items

def get_top15(word, vocab_list, func):
    sim_vec = [(w, func(word,w)) for w in vocab_list]
    sim_vec = sorted(sim_vec, key=lambda t: t[1])
    return jsonify(
            word = word,
            similar = [i for i in reversed(sim_vec[-15:])],
            different = sim_vec[:15])
