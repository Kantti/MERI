import re, sys, os, json
import wget
from resources import *




def download_collocation_vectors(data):

    for dec in data:
        i = 0
        if not os.path.exists("../data/collocations/"+str(dec)):
            os.mkdir("../data/collocations/"+str(dec))
        times = []
        for word in data[dec]:
            i += 1
            PATH = "../data/collocations/"+str(dec)+"/"+word+".json"
            if len(word) > 3 and not os.path.exists(PATH) and not in_process(word, dec):
                T = time.time()
                print(i, "/", len(data[dec]))
                query_word(word, dec, 10000)
                with open(PATH, "r", encoding="utf-8") as f:
                    vector = get_freq_list(json.load(f)["kwic"])
                    vector = filter_freq_list(vector, 1)
                with open(PATH, "w", encoding="utf-8") as f:
                    json.dump(vector, f)
                times.append(time.time()-T)
                print(sum(times)/len(times))

def in_process(WORD, DEC):
    for P in os.listdir("../data/collocations/"+str(DEC)+"/"):
        if WORD in P and P.endswith("tmp"):
            return True
    return False

if __name__ == "__main__":
    KEY = sys.argv[1]
    data = open_data(KEY)
    data = clean_data(data, KEY)
    data = parse_data_by_decades(data)
    data = {x:get_freq_list(data[x]) for x in data}
    data = {x:filter_freq_list(data[x], 3) for x in data}
    download_collocation_vectors(data)




