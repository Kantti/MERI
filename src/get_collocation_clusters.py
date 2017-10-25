import re, sys, os, json
from resources import *
import pandas

def get_collocation_matrix(DEC):
    PATH = "../data/collocations/"+str(DEC)+"/"
    df = {}
    for F in os.listdir(PATH):
        if F.endswith("json"):
            print(os.listdir(PATH).index(F), "/", len(os.listdir(PATH)))
            with open(PATH+F, "r", encoding="utf-8") as f:
                data=json.load(f)["kwic"]
            data = get_freq_list(data)
            df[F.replace(".json", "")] = data

    df = pandas.DataFrame(df)
    return df


if __name__ == "__main__":

    df = get_collocation_matrix(1890)
    print(df)
