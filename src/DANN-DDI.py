# DANN-DDI
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
import csv
import random
import sys
sys.path.append("..")
import graph, sdne
import keras
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
import datetime
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Activation, BatchNormalization
from keras.utils import to_categorical

event_num=2
droprate=0.4
vector_size=128


def DNN():
    train_input1 = Input(shape=(vector_size * 5,), name='Inputlayer1')
    train_input2 = Input(shape=(vector_size * 5,), name='Inputlayer2')

    # Attention Neural Network
    train_input = keras.layers.Concatenate()([train_input1, train_input2])

    attention_probs = Dense(vector_size * 5,activation='relu',use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros', name='attention1')(train_input)

    att = Dense(vector_size * 5,activation='softmax', kernel_initializer='random_uniform', name='attention')(attention_probs)

    vec = keras.layers.Multiply()([train_input1, train_input2])   
    attention_mul = keras.layers.Multiply()([vec, att])

    # Deep neural network classifier
    train_in = Dense(4096, activation='relu', name='FullyConnectLayer1')(attention_mul)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)

    train_in = Dense(2048, activation='relu', name="FullyConnectLayer2")(train_in)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)

    train_in = Dense(1024, activation='relu', name="FullyConnectLayer3")(train_in)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)

    train_in = Dense(512, activation='relu', name="FullyConnectLayer4")(train_in)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)

    train_in = Dense(256, activation='relu', name="FullyConnectLayer5")(train_in)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)

    train_in = Dense(128, activation='relu', name="FullyConnectLayer6")(train_in)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)

    train_in = Dense(64, activation='relu', name="FullyConnectLayer7")(train_in)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)

    train_in = Dense(event_num, name="SoftmaxLayer")(train_in)

    out = Activation('softmax', name="OutputLayer")(train_in)

    model = Model(inputs=[train_input1,train_input2],outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def calculate_metric_score(real_labels,predict_score):
    # Evaluate the prediction performance
    precision, recall, pr_thresholds = precision_recall_curve(real_labels, predict_score)
    aupr_score = auc(recall, precision)
    all_F_measure = np.zeros(len(pr_thresholds))
    for k in range(0, len(pr_thresholds)):
       if (precision[k] + recall[k]) > 0:
           all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
       else:
           all_F_measure[k] = 0
    print("all_F_measure: ")
    print(all_F_measure)
    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]
    fpr, tpr, auc_thresholds = roc_curve(real_labels, predict_score)
    auc_score = auc(fpr, tpr)

    f = f1_score(real_labels, predict_score)
    print("F_measure:"+str(all_F_measure[max_index]))
    print("f-score:"+str(f))
    accuracy = accuracy_score(real_labels, predict_score)
    precision = precision_score(real_labels, predict_score)
    recall = recall_score(real_labels, predict_score)
    print('results for feature:' + 'weighted_scoring')
    print(    '************************AUC score:%.3f, AUPR score:%.3f, precision score:%.3f, recall score:%.3f, f score:%.3f,accuracy:%.3f************************' % (
        auc_score, aupr_score, precision, recall, f, accuracy))
    results = [auc_score, aupr_score, precision, recall,  f, accuracy]

    return results


def cross_validation(drug_drug_matrix, CV_num):
    # 3-folds or 5-folds cross validation
    results = []
    link_number = 0
    link_position = []
    nonLinksPosition = []

    for i in range(0, len(drug_drug_matrix)):
        for j in range(i + 1, len(drug_drug_matrix)):
            if drug_drug_matrix[i, j] == 1:
                link_number = link_number + 1
                link_position.append([i, j])
            elif drug_drug_matrix[i, j] == 0 and np.sum(drug_drug_matrix[i,:],axis=0) > 0 and np.sum(drug_drug_matrix[:,j],axis=0) > 0:
                nonLinksPosition.append([i, j])

    link_position = np.array(link_position)
    print("link_position:" + str(len(link_position)))
    nonLinksPosition = np.array(nonLinksPosition)
    print("nonLinksPosition:" + str(len(nonLinksPosition)))

    index = np.arange(0, len(link_position))
    random.shuffle(index)

    fold_num = len(link_position) // CV_num
    print(fold_num)

    for CV in range(0, CV_num):
        print('*********round:' + str(CV) + "**********\n")
        starttime = datetime.datetime.now()

        #  Build the drug-drug interaction network
        g = graph.Graph()
        g.read_edgelist('../data/dataset/drug_drug.txt')
        print(g.G.number_of_edges())

        test_index = index[(CV * fold_num):((CV + 1) * fold_num)]
        train_index  = np.setdiff1d(index, test_index)

        test_index.sort()
        train_index.sort()
        print(len(test_index)+len(train_index))

        testPosition = np.array(link_position)[test_index]
        print(testPosition)
        trainPosition = np.array(link_position)[train_index]
        print(trainPosition)
        print("testPosition:" + str(len(testPosition)))
        print("trainPosition:" + str(len(trainPosition)))

        # Remove the test_links in the network
        for i in range(0, len(testPosition)):
            if drug_drug_matrix[testPosition[i, 0]][testPosition[i, 1]] == 1:
                g.G.remove_edge(str(testPosition[i, 0] + 1), str(testPosition[i, 1] + 1))
        print(g.G.number_of_edges())

        # Obtain representation vectors by SDNE
        print("Test Begin")
        model = sdne.SDNE(g, [1000, 128],)
        print("Test End")

        data = pd.DataFrame(model.vectors).T
        data.to_csv('../data/embeddings/d_embeddings.csv', header=None)

        model_s=loadmodel('../data/embeddings/s_embeddings.csv')
        model_t=loadmodel('../data/embeddings/t_embeddings.csv')
        model_e=loadmodel('../data/embeddings/e_embeddings.csv')
        model_p=loadmodel('../data/embeddings/p_embeddings.csv')
        I1 = []
        with open('../data/embeddings/d_embeddings.csv', "rt", encoding='utf-8')as csvfile1:
            reader = csv.reader(csvfile1)
            for i in reader:
                I1.append(i[0])
        I1.sort()

        # Concatenate of representation vectors generated by five drug feature networks
        E=np.zeros((841, 640), float)
        for i in I1:
            E[int(i)-1][0:128]=model_s[int(i)-1]
            E[int(i)-1][128:256]=model_t[int(i)-1]
            E[int(i)-1][256:384]=model_e[int(i)-1]
            E[int(i)-1][384:512]=model_p[int(i)-1]
            E[int(i)-1][512:640]=model.vectors[str(i)]

        # Training set
        X_train1 = []
        X_train2 = []
        Y_train = []
        trainPosition = np.concatenate((np.array(trainPosition), nonLinksPosition), axis=0)

        for i in range(0, len(trainPosition)):
            X_train1.append(E[(trainPosition[i, 0])])
            X_train2.append(E[(trainPosition[i, 1])])
            Y_train.append(drug_drug_matrix[trainPosition[i, 0], trainPosition[i, 1]])

        X_train1 = np.array(X_train1)
        X_train2 = np.array(X_train2)
        Y_train = np.array(Y_train)
        Y_train = to_categorical(Y_train, 2)

        dnn = DNN()
        dnn.fit([X_train1,X_train2],Y_train,batch_size=128, epochs=150, shuffle=True, verbose=1)

        # Test set
        X_test1= []
        X_test2= []
        Y_test = []
        testPosition = np.concatenate((np.array(testPosition), nonLinksPosition), axis=0)

        for i in range(0, len(testPosition)):
            X_test1.append(E[(testPosition[i, 0])])
            X_test2.append(E[(testPosition[i, 1])])
            Y_test.append(drug_drug_matrix[testPosition[i, 0], testPosition[i, 1]])

        X_test1 = np.array(X_test1)
        X_test2 = np.array(X_test2)
        y_pred_label = dnn.predict([X_test1,X_test2])

        y_pred_label = np.argmax(y_pred_label, axis=1)
        y_pred_label = np.array(y_pred_label).tolist()

        results.append(calculate_metric_score(Y_test, y_pred_label))

        endtime = datetime.datetime.now()
        print(endtime - starttime)
    return results


def Interections():
    # Generate drug_drug interactions matrix
    M = []  
    N = [] 
    dic = {}  

    with open('../data/dataset/Node_Codes.csv', "rt", encoding='utf-8')as csvfile:
        reader = csv.reader(csvfile)
        for i in reader:
            M.append(i[0])
            N.append(i[1])

        for i in range(len(M)):
            dic[M[i]] = N[i]

    D = []
    I = []

    with open('../data/dataset/Drug_Information.csv', "rt", encoding='utf-8')as csvfile:
        reader = csv.reader(csvfile)
        for i in reader:
            D.append(i[0])
            I.append(i[1])

    DDI = np.zeros((len(D), len(D)), int)

    for i in range(len(D)):
        for j in I[i].split('|'):
            if not j.strip() == '' and j in M:
                DDI[int(dic[D[i]]) - 1][int(dic[j]) - 1] = 1
    return DDI


def loadmodel(path):
    # Load the files of representation vectors generated before
    Emb = pd.read_csv(path, header=None)
    features = list(Emb.columns)
    Emb = np.array(Emb[features])
    return Emb


def main():
    adj = Interections()
    results = cross_validation(adj, 5)
    print(results)
    results = np.array(results)
    print(results.sum(axis=0) / 5)


if __name__ == '__main__':
    main()
