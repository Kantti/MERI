import re, sys, os, json
import pandas
from resources import *




def get_year_matrix(data):
    res = {}
    for row in data:
        year = row["structs"]["text_issue_date"][-4:]
        if year not in res:
            res[year] = {}
        for L in get_lemmas(row):
            if L not in res[year]:
                res[year][L] = 1
            else:
                res[year][L] += 1

    return res


if __name__ == "__main__":

    data = open_data(sys.argv[1])
    matrix = get_year_matrix(data)
    df = pandas.DataFrame(matrix)
    df = df[df.sum(axis=1) > 4]
    df = df.transpose()
    df = df.fillna(0)
    df = df.div(df.sum(axis=1), axis=0)
    print(df)
    plot_DF(df, 2, "cosine")
    
