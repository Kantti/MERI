from resources import *

def get_token_cluster_matrix(data, clusters):

    res = numpy.zeros(shape=(len(data), max(clusters.values())))

    for i in range(len(data)):
        for L in get_lemmas(data[i]):
            if L in clusters:
                res[i,clusters[L]-1] += 1

    print(res)
    return res

def get_aggregated_clusters(data, clusters):

    res = {i:0 for i in range(1,max(clusters.values())+1)}

    for row in data:
        for L in get_lemmas(row):
            if L in clusters:
                res[clusters[L]] += 1

    return res

if __name__ == "__main__":

    DEC = sys.argv[1]
    WORD = sys.argv[2]

    data = open_decade(WORD, DEC)
    data = parse_data_by_publ_id(data)
    paper_data = get_paper_names_for_ids()
    sizes = {key:len(data[key]) for key in data}
    sizes = {paper_data[x.lower()]:sizes[x]/max(sizes.values())*100 for x in sizes}
    
    clusters = get_clusters_from_distance_matrix(DEC, "cosine", 2.0)
    print("found", max(clusters.values()), "clusters")
    df = {paper_data[key.lower()]:get_aggregated_clusters(data[key], clusters) for key in data}
    df = {key:df[key] for key in df if sum(df[key].values()) > 0}
    df = pandas.DataFrame(df).transpose().fillna(0)
#    print(df.loc["Työmiehen Ystävä"])
#    print(df.loc["Satakunta"])
    if sys.argv[3] == "mds":
        plot_DF(df, 2, "cosine", sizes=sizes)
    if sys.argv[3] == "h":
        plot_dendrogram(df, "cosine")


    

