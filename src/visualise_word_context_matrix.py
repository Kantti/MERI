from resources import *

DROPLIST = ["ettäei", "heijari", "vuositarkastus", "aljetaka", "taupunka", "toimipito"]


if __name__ == "__main__":

    WORD = sys.argv[1]
    DEC = sys.argv[2]
    METHOD = sys.argv[3]
    METRIC = sys.argv[4]
    data = open_data(WORD)
    data = clean_data(data, WORD)
    data = parse_data_by_decades(data)[int(DEC)]
    data = get_freq_list(data)
    data = filter_freq_list(data, 3)

    with open("../data/collocations/"+str(DEC)+"_distance_matrix_"+METRIC+".json", "r", encoding="utf-8") as f:
        dist = json.load(f)
    
    dist = {x:{y:dist[x][y] for y in dist[x] if y in data and y not in DROPLIST} for x in dist if x in data and x not in DROPLIST}
    dist = pandas.DataFrame(dist)
    if METHOD == "mds":
        clusters = get_clusters(dist, 2.0)
        read_clusters(clusters)
        
        plot_DF(dist, 2, False, clusters)
    elif METHOD == "hierarch":
        plot_dendrogram(dist, False)

    
