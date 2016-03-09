from app.gloveSim import gloveSim

from app.utils import read_csv, lemmatize_an_idea
import random
import pickle
import cProfile
import timeit

TOPICS = {
    'weddingTheme': 'topicWords/wedding_themes_collapsed_5.csv',
    'weddingProp': 'topicWords/wedding_props_collapsed_5.csv'

}


with open('app/theme_dict_set.p') as f:
    theme_dict_set = pickle.load(f)

with open('app/prop_dict_set.p') as f:
    prop_dict_set = pickle.load(f)

topics = {}
for k,v in TOPICS.iteritems():
    topics[k] = [(w[0], w[1].lower()) for w in read_csv(v)]

def test_glove_func(word, topic, operation):
    func = gloveSim
    this_dict_set = theme_dict_set if topic=='weddingTheme' else prop_dict_set
    vocab_list = this_dict_set['words']
    sim_vec = [{'id':w[0], 'text':w[1], 'similarity':func(word,w[1])}
        for w in vocab_list if ' '.join(lemmatize_an_idea(w[1])) != ' '.join(lemmatize_an_idea(word)) and func(word,w[1]) > -100]

    sim_vec = sorted(sim_vec, key=lambda t: t['similarity'])
    for_sim = [t for t in sim_vec if t['similarity'] < 0.5]
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

    # print(word)
    # print(similar_sets)
    # print(different_sets)

if __name__ == '__main__':
    # cProfile.run("test_glove_func('cowboy', 'weddingTheme', 'similar')")
    # cProfile.run("test_glove_func('cowboy', 'weddingTheme', 'different')")
    t = timeit.Timer("test_glove_func('cowboy', 'weddingTheme', 'similar')", setup="from __main__ import test_glove_func")
    t2 = timeit.Timer("test_glove_func('cowboy', 'weddingTheme', 'different')", setup="from __main__ import test_glove_func")
    t3 = timeit.Timer("test_glove_func('hawaii', 'weddingTheme', 'different')", setup="from __main__ import test_glove_func")
    print t.timeit(1)
    print t2.timeit(1)
    print t3.timeit(1)
    # timeit.timeit("test_glove_func('cowboy', 'weddingTheme', 'different')")







