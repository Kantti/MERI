import re, sys, json, os
import pandas
from resources import *
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import pairwise_distances
from matplotlib import pyplot as plt


def open_wordlist():
    with open("../resources/vaivainen_forms.json", "r", encoding="utf-8") as f:
        return json.load(f)


def get_words(kwic, msd=False):
    res = {}
    for row in kwic:
        i = row["match"]["start"]

        if msd:
            word = parse_msd(row["tokens"][i]["msd"])
            if "NUM" not in word:
                word["NUM"] = "XXX"
            if "POSS" in word:
                end = "_POSS"
            else:
                end = ""
            word = word["NUM"]+"_"+word["CASE"]+end
            
        else:
            word = fix_word(row["tokens"][i]["word"], i, row)

        if word not in res:
            res[word] = 1
        else:
            res[word] += 1

    return res

    

def filter_clits(words):
    keys = list(words.keys())
    for W in keys:
        ends = ends_with_clit(W)
        if ends:
            if ends not in words:
                words[ends] = words[W]
            else:
                words[ends] += words[W]
            del words[W]

    return words

def get_yearly_matrix(kwic, wordlist, comps):
    res = {}
    for row in kwic:

        i = row["match"]["start"]
        word = fix_word(row["tokens"][i]["word"], i, row)
        if word == "vaivaiselta":
            print(word in wordlist, word in comps)
        if word in wordlist or word in comps:
            year = row["structs"]["text_issue_date"][-4:]
            if word not in wordlist:
                word = "waiwais-"
            else:
                word = wordlist[word]
            if year not in res:
                res[year] = {}
            if word not in res[year]:
                res[year][word] = 1
            else:
                res[year][word] += 1

    return res

def get_graphs(df):
    
    res = pandas.DataFrame()
    compound = df.loc['waiwais-']
    compound = compound.div(df.sum(axis=0))
    df = df.drop("waiwais-")
    print(df.index)
    res['sg inner'] = df.loc[[x for x in df.index if x.split("_")[1] in INNER_CASES and "Sg" in x]].sum(axis=0)
    res['sg outer'] = df.loc[[x for x in df.index if x.split("_")[1] in OUTER_CASES and "Sg" in x]].sum(axis=0)
    res['sg syntactic'] = df.loc[[x for x in df.index if x.split("_")[1] in SYNT_CASES and "Sg" in x]].sum(axis=0)
    res['pl inner'] = df.loc[[x for x in df.index if x.split("_")[1] in INNER_CASES and "Pl" in x]].sum(axis=0)
    res['pl outer'] = df.loc[[x for x in df.index if x.split("_")[1] in OUTER_CASES and "Pl" in x]].sum(axis=0)
    res['pl syntactic'] = df.loc[[x for x in df.index if x.split("_")[1] in SYNT_CASES and "Pl" in x]].sum(axis=0)
    res = res.div(res.sum(axis=1), axis=0)
#    res['compound'] = compound
    print(res)
    res.plot()
    plt.show()

def get_comparing_matrix(words):
    return {W:get_words(open_data(W), msd=True) for W in words}


def get_fixed_yearly_matrix(W):

    raw_data = open_data(W)
    data = get_words(raw_data)
    data = filter_clits(data)
    wordlist = open_wordlist()
    comp_keys = {x:data[x] for x in data if x not in wordlist and data[x] > 5}
    return get_yearly_matrix(raw_data, wordlist, comp_keys)

def get_mixed_vector(A, B, a, b):
    return pandas.concat([A.multiply(a),B.multiply(b)], axis=1).sum(axis=1).div(a+b).fillna(0)
    

def optimize_combination(A,B,C):
    MIN = 1.0
    for i in range(1,1000):
        df = pandas.concat((A,get_mixed_vector(B,C,i,1000-i)), axis=1)
        df = df.div(df.sum(axis=1), axis=0)
        dist = pairwise_distances(df, metric="euclidean")[0,1]
        if dist < MIN:
            print(i, dist)
            MIN = dist
            res = i

    return res


def compare(WORDS):

    matrix = get_comparing_matrix(WORDS)

    df = pandas.DataFrame(matrix)
    df = df.fillna(0)
    df = df.transpose()
    mix = optimize_combination(df.loc['poika'], df.loc['tytto'], df.loc['tytar'])

    df.loc['combined'] = get_mixed_vector(df.loc['tytto'], df.loc['tytar'], mix, 1000-mix)

    df = df.div(df.sum(axis=1), axis=0)
    dist = pairwise_distances(df, metric="euclidean")
    for i in range(len(df.index)):
        for j in range(len(df.index)):
            print(df.index[i], df.index[j], dist[i,j])



if __name__ == "__main__":

    WORDS = sys.argv[1].split(",")
    
    if len(WORDS) == 1:
        matrix = get_fixed_yearly_matrix(WORDS[0])
        df = pandas.DataFrame(matrix)
        df = df.transpose()
        df = df.fillna(0)
        df = df.div(df.sum(axis=1), axis=0)
        print(df)
        sys.exit()
        plot_DF(df, 1, "cosine")
    

    else:
        compare(WORDS)




