from resources import *
import time

def open_vector(W, DEC):

    with open("../data/collocations/"+str(DEC)+"/"+W+".json", "r", encoding="utf-8") as f:
        return json.load(f)

def dicts_to_DF(A, B):
    T = time.time()
    df = {"A":A, "B":B}
    df = pandas.DataFrame(df)
    df = df.transpose()
    df = df.fillna(0)
    print("dicts to DF", time.time()-T)
    return df

def dicts_to_array(A,B):
    keys = list(set(list(A.keys())+list(B.keys())))
    A_zeros = [0 for i in range(len(keys))]
    B_zeros = [0 for i in range(len(keys))]
    Z = numpy.zeros(shape=(2,len(keys)))
    A_sum = sum(A.values())
    B_sum = sum(B.values())
    
    for i in range(len(keys)):
        if keys[i] in A:
            Z[0,i] = A[keys[i]]/A_sum
        if keys[i] in B:
            Z[1,i] = B[keys[i]]/B_sum
#    Z = numpy.array([A_zeros,B_zeros])

#        Z[0,i] = A[keys[i]]/A_sum if keys[i] in A else 0 for i in range(len(keys))]
#        Z[1,i] = [B[keys[i]]/B_sum if keys[i] in B else 0 for i in range(len(keys))]
    return Z


def get_distances(W, DEC, keys, done):
    
    main_vector = open_vector(W, DEC)
    res = {}
    D_times = []
    A_times = []
    F_times = []
    res_cos = numpy.zeros(shape=(1,len(keys)))
    res_euc = numpy.zeros(shape=(1,len(keys)))
    T = time.time()
    for i in range(len(keys)):
        T = time.time()
        if keys[i] not in done:
            comp_vector = open_vector(keys[i], DEC)
            try:
                if sum(comp_vector.values()) == 0:
                    print(keys[i])
                else:
#        df = dicts_to_DF(main_vector,comp_vector)
                    AT = time.time()
                    np = dicts_to_array(main_vector,comp_vector)
                    A_times.append(time.time()-AT)
#        T = time.time()
#        dist_df = pairwise_distances(df, metric="cosine")
#        print("distance calc df", time.time()-T)
#        T = time.time()
                    dist_cos = pairwise_distances(np, metric="cosine")
                    dist_euc = pairwise_distances(np, metric="euclidean")
                    res_cos[0,i] = dist_cos[0,1]
                    res_euc[0,i] = dist_euc[0,1]

#        print("distance calc np", time.time()-T)
#        print(dist_df, dist_np)
            except:
                TypeError
                print(keys[i])
                sys.exit()
        F_times.append(time.time()-T)
    print("array times:", sum(A_times)/len(A_times))
    print("full timing:", sum(F_times)/len(F_times))
    print("all in all:", sum(F_times))
    return res_cos, res_euc

def mirror_distance_matrix(matrix):

    for x in matrix:
        for y in matrix:
            if matrix[x][y] == 0.0 and matrix[y][x] != 0.0:
                matrix[x][y] = matrix[y][x]
    return matrix

if __name__ == "__main__" :

    DEC = sys.argv[1]
    RES_PATH_COS = "../data/collocations/"+str(DEC)+"_distance_matrix_cosine.json"
    RES_PATH_EUC = "../data/collocations/"+str(DEC)+"_distance_matrix_euclidean.json"
    if os.path.exists(RES_PATH_COS):
        with open(RES_PATH_COS, "r", encoding="utf-8") as f:
            matrix = json.load(f)
        matrix = mirror_distance_matrix(matrix)
        with open(RES_PATH_COS, "w", encoding="utf-8") as f:
            json.dump(matrix, f)
    else:
        keys = [x.replace(".json", "") for x in os.listdir("../data/collocations/"+str(DEC)+"/")]
        done = []
        res_cos = numpy.zeros(shape=(len(keys), len(keys)))
        res_euc = numpy.zeros(shape=(len(keys), len(keys)))
        for i in range(len(keys)):
            cos, euc = get_distances(keys[i], DEC, keys, done)
            res_cos[i,:] = cos
            res_euc[i,:] = euc
            done.append(keys[i])
            print("round", i, len(keys))

        res_cos = {keys[i]:{keys[j]:res_cos[i,j] for j in range(len(keys))} for i in range(len(keys))}
        res_cos = mirror_distance_matrix(res_cos)
        with open(RES_PATH_COS, "w", encoding="utf-8") as f:
            json.dump(res_cos, f)
        
        res_euc = {keys[i]:{keys[j]:res_euc[i,j] for j in range(len(keys))} for i in range(len(keys))}
        res_euc = mirror_distance_matrix(res_euc)
        with open(RES_PATH_EUC, "w", encoding="utf-8") as f:
            json.dump(res_euc, f)

