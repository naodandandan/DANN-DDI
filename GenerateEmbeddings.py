# Generate Embeddings from drug feature networks
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append("..")
import graph, sdne
from keras.models import Model


def generateE(path1,path2):
    g = graph.Graph()
    g.read_edgelist(path1)
    print(g.G.number_of_edges())

    # Generate representation vectors by SDNE
    print("Test Begin")
    model = sdne.SDNE(g, [1000, 128],)
    print("Test End")

    drug_id = {}
    with open("../data/dataset/id_drug.txt", encoding='utf-8', errors='ignore') as f1:
        for line in f1:
            toks = line.strip().split(",")
            if len(toks) == 2:
                drug_id[toks[1]] = toks[0]

    f2 = open(path2, 'wt', encoding='utf-8')
    for a, b in drug_id.items():
        arr = model.vectors[str(b)]
        for i in range(len(arr)):
            f2.write(str(arr[i]))
            if i + 1 != len(arr):
                f2.write(',')
        f2.write('\n')


def main():
    # drug-substructure network
    print("******************************************generate s_embeddings*******************************************")
    generateE("../data/dataset/drug_structure.txt", "../data/embeddings/s_embeddings.csv")

    # drug-substructure network
    print("******************************************generate t_embeddings*******************************************")
    generateE("../data/dataset/drug_target.txt", "../data/embeddings/t_embeddings.csv")

    # drug-enzyme network
    print("******************************************generate e_embeddings*******************************************")
    generateE("../data/dataset/drug_enzyme.txt", "../data/embeddings/e_embeddings.csv")

    # drug-pathway network
    print("******************************************generate p_embeddings*******************************************")
    generateE("../data/dataset/drug_path.txt", "../data/embeddings/p_embeddings.csv")


if __name__ == '__main__':
    main()
