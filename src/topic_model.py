import re, sys, os, json
import statistics
from resources import *
import lda
import numpy as np


def data_to_dtm(clear=True, dec=False):

    if not clear:
        with open("../tmp/all_words.json", "r", encoding="utf-8") as f:
            all_words = json.load(f)
        return all_words, np.load("../tmp/dtm.csv.npy")
        
        
    data = open_data("waiwainen")
    if dec:
        data = parse_data_by_decades(data)[dec]
    data = clean_data(data)
    freqs = {}
    for row in data:
        for W in get_lemmas(row):
            if W not in freqs:
                freqs[W] = 1
            else:
                freqs[W] += 1

    with open("../tmp/freqs.json", "w", encoding="utf-8") as f:
        json.dump(freqs, f)
    all_words = [x for x in freqs if freqs[x] > 1]
    print(len(data), len(all_words))
    x = len(data)
    y = len(all_words)
    res = np.zeros(shape=(x, y), dtype=int)
    print("empty matrix initialized")
    for i in range(len(data)):
        for W in get_lemmas(row):
            if W in all_words:
                j = all_words.index(W)
                res[i,j] += 1
    print("matrix built")
    np.save("../tmp/dtm.csv", res)
    with open("../tmp/all_words.json", "w", encoding="utf-8") as f:
        json.dump(all_words, f)

    return all_words, res

def clean_data(data):
    return [x for x in data if fix_matched_word(x) in open_wordlist()]
    
def get_topics(LDA, all_words):

    res = {x:[0 for y in range(len(LDA.components_))] for x in all_words}
    for i in range(len(LDA.components_)):
        for j in range(len(LDA.components_[i,:])):
            res[all_words[j]][i] = LDA.components_[i,j]

    topics = [{} for x in range(len(LDA.components_))]

    for W in res:
        topics[res[W].index(max(res[W]))].update({W:max(res[W])/statistics.mean(res[W])})

    for T in topics:
        print(len(T))


if __name__ == "__main__":

    if "clear" in sys.argv:
        clear = True
    else:
        clear = False
    all_words, data = data_to_dtm(clear=clear, dec=1990)
    LDA = lda.lda.LDA(5, alpha=5.0)
    LDA.fit(data)
    get_topics(LDA, all_words)
