import re,sys,os,json, wget, time, csv
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import pandas, numpy
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import randomcolor

INNER_CASES = ("Ela", "Ill", "Ine")
OUTER_CASES = ("Abl", "All", "Ade")
SYNT_CASES = ("Nom", "Par", "Gen")
ABSTRACT = ("Tra", "Ess")
H_GROUP = ("Pl_Ela", "Sg_Gen", "Pl_Nom", "Pl_All", "Sg_Nom", "Sg_Par")

NO_CORPUS = [1828, 1843]

def parse_data_by_decades(data):

    res = {x:[] for x in range(1800,2000, 10)}

    for row in data:
        dec=int(row["structs"]["text_issue_date"][-4:][:-1]+"0")
        res[dec].append(row)

    return res


def parse_data_by_publ_id(data):

    res = {}

    for row in data:
        key = row["structs"]["text_publ_id"]
        if key not in res:
            res[key] = [row]
        else:
            res[key].append(row)
    return res

def turn_html(W):
    W = W.replace("ä", "%C3%A4")
    W = W.replace("ö", "%C3%B6")
    W = W.replace("Ö", "%C3%96")
    W = W.replace("Ä", "%C3%84")
    
    W = W.replace("å", "a")
    return W

def get_freq_list(data):
    res = {}
    for row in data:
        words = get_lemmas(row)
        for W in words:
            if W not in res:
                res[W] = 1
            else:
                res[W] += 1
    return res

def open_data(code):
    with open("../data/"+code+".json", "r", encoding="utf-8") as f:
        return json.load(f)["kwic"]

def get_lemmas(row):
    return [row["tokens"][i]["lemma"] for i in range(len(row["tokens"])) if i != row["match"]["start"] and row["tokens"][i]["lemma"].isalpha()]


def filter_freq_list(F, i):
    return {x:F[x] for x in F if F[x] > i}

def parse_msd(msd):
    return {x.split("_")[0]:x.split("_")[1] for x in msd.split("|")}

def fix_word(word, i, row):
        
    if word.endswith("-") and i < len(row["tokens"])-1:
        word = word[:-1]+row["tokens"][i+1]["word"]
    word = word.lower()
    word = word.replace("w", "v")
    if word[5] != "i":
        word = word[:5]+"i"+word[5:]

    return word

def fix_matched_word(row):
    i = row["match"]["start"]
    word = row["tokens"][i]["word"]
    word = fix_word(word, i, row)
    clit = ends_with_clit(word)
    if clit:
        return clit
    return word

def ends_with_clit(word): 
    for C in ("kin", "kaan", "kään", "han", "hän", "ko", "kö"): 
        if word.endswith(C): 
            return word[:-len(C)] 
                                 
    return False 

def get_paper_names_for_ids():
    with open("../resources/newspapers-utf8.csv", "r", encoding="utf-8") as f:
        cread = csv.reader(f, delimiter=",")
        data = [x for x in cread]

    return {x[0].lower():x[3] for x in data}


def plot_DF(df, N, metric, annotate=True, clusters=False, sizes=False):

    if metric:
        if metric != "cosine":
            df = df.div(df.sum(axis=0), axis=1)
        dist = pairwise_distances(df, metric=metric)
    else:
        dist = df
    mds = MDS(dissimilarity="precomputed", n_components=N)
    pos = mds.fit(dist).embedding_

        

    
    if N == 1:

        for x,y in zip(df.index, pos):
            plt.scatter(x,y)
            
    elif N == 2:
        if clusters:
            colors = get_colors(clusters)

        for l,x,y in zip(df.index, pos[:,0], pos[:,1]):
            
            if sizes:
                S=sizes[l]
            else:
                S=10

            if clusters:
                plt.scatter(x,y,c=colors[l],s=S)
            else:
                plt.scatter(x,y,s=S)
            if annotate:
                plt.annotate(l, xy=(x,y))

    plt.show()

def get_colors(clusters):
    rand_color = randomcolor.RandomColor()
    colors = rand_color.generate(count=len(clusters))
    return {x:colors[clusters[x]] for x in clusters}


def read_clusters(clusters):

    C = [[] for i in range(max(clusters.values()))]

    for X in clusters:
        C[clusters[X]-1].append(X)
    res = []
    for i in range(max([len(x) for x in C])):
        line = []
        for j in range(len(C)):
            if i < len(C[j]):
                line.append(C[j][i])
            else:
                line.append("-")

        res.append("\t".join(line))

    with open("../tmp/cluster_view.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(res))

def get_clusters_from_distance_matrix(DEC, METRIC, DIST):

    with open("../data/collocations/"+str(DEC)+"_distance_matrix_"+METRIC+".json", "r", encoding="utf-8") as f:
        wdm = json.load(f)

    return get_clusters(pandas.DataFrame(wdm), DIST)


def open_decade(KEY, DEC):
    res = open_data(KEY)
    res = parse_data_by_decades(res)[int(DEC)]
    return clean_data(res, KEY)

def open_wordlist():
    with open("../resources/vaivainen_forms.json", "r", encoding="utf-8") as f:
        return json.load(f)


def plot_dendrogram(df, metric):
    if metric != "cosine":
        df = df.div(df.sum(axis=1), axis=0)

    Z = linkage(df, method="ward")
    dendrogram(Z, labels=df.index, leaf_rotation=90, leaf_font_size=10)
    plt.show()

def get_clusters(df, MAX_D):

    Z = linkage(df, method="ward")
    clusters = fcluster(Z, MAX_D, criterion="distance")
    return {df.index[i]:clusters[i] for i in range(len(df.index))}


def clean_data(data, word):
    if word == "waiwainen":
        return [x for x in data if fix_matched_word(x) in open_wordlist()]
    return data


def query_word(W, DEC, MAX):
    PATH = "../data/collocations/"+str(DEC)+"/"+W+".json"
    if not os.path.exists(PATH):
        corpora = "%2C".join(["KLK_FI_"+str(i) for i in range(DEC, DEC+10) if i not in NO_CORPUS])
        query = "https://korp.csc.fi/cgi-bin/korp.cgi?command=query&defaultcontext=paragraph&show=lemma&show_struct=text_issue_date&cache=true&start=0&end="+str(MAX)
        query+="&corpus="+corpora
        query += "&context=&incremental=true&sort=random&random_seed=123456"
        query += "&cqp=%5Blemma+%3D+%22"+turn_html(W)+"%22%5D&defaultwithin=sentence&within="
        try:
            f = wget.download(query, out=PATH)
        except:
            UnicodeEncodeError
            print(turn_html(W))
            sys.exit()
