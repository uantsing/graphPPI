import os
import numpy as np
import random
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, auc
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
import torch
import ipdb

def print_file(str_, save_file_path=None):
    print(str_)
    if save_file_path != None:
        f = open(save_file_path, 'a')
        print(str_, file=f)

class Metrictor_PPI:
    def __init__(self, pre_y, truth_y, true_prob, is_binary=False):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        self.pre = np.array(pre_y).squeeze()
        self.tru = np.array(truth_y).squeeze()
        self.true_prob = np.array(true_prob).squeeze()
        if is_binary:
            length = pre_y.shape[0]
            for i in range(length):
                if pre_y[i] == truth_y[i]:
                    if truth_y[i] == 1:
                        self.TP += 1
                    else:
                        self.TN += 1
                elif truth_y[i] == 1:
                    self.FN += 1
                elif pre_y[i] == 1:
                    self.FP += 1
            self.num = length

        else:
            N, C = pre_y.shape
            for i in range(N):
                for j in range(C):
                    if pre_y[i][j] == truth_y[i][j]:
                        if truth_y[i][j] == 1:
                            self.TP += 1
                        else:
                            self.TN += 1
                    elif truth_y[i][j] == 1:
                        self.FN += 1
                    elif truth_y[i][j] == 0:
                        self.FP += 1
            self.num = N * C
    

    def show_result(self, is_print=False, file=None):
        self.Accuracy = (self.TP + self.TN) / (self.num + 1e-10)
        self.Precision = self.TP / (self.TP + self.FP + 1e-10)
        self.Recall = self.TP / (self.TP + self.FN + 1e-10)
        self.F1 = 2 * self.Precision * self.Recall / (self.Precision + self.Recall + 1e-10)
        aupr_entry_1 = self.tru
        aupr_entry_2 = self.true_prob
        aupr = np.zeros(13)
        for i in range(13):

            precision, recall, _ = precision_recall_curve(aupr_entry_1[:,i], aupr_entry_2[:,i])
            aupr[i] = auc(recall,precision)
            
        self.Aupr = aupr

        if is_print:
            print_file("Accuracy: {}".format(self.Accuracy), file)
            print_file("Precision: {}".format(self.Precision), file)
            print_file("Recall: {}".format(self.Recall), file)
            print_file("F1-Score: {}".format(self.F1), file)

class Multigraph2Big:

    def __init__(self, p_x_all, p_edge_all) -> None:

        """
        p_x_all, p_edge_all 是 full graph 的所有节点的特征
        """

        assert len(p_x_all) == len(p_edge_all)

        self.graph_list = []
        f = open('error_p_num.txt', 'w')

        for i in range(len(p_x_all)):

            try:
                self.graph_list.append(Data(x=torch.Tensor(p_x_all[i]), 
                                  edge_index=torch.LongTensor(p_edge_all[i]).transpose(1, 0)))
            except:
                print('{}, {}'.format(p_edge_all[i], i))
                f.write(str(i)+'\n')



    def process(self, node_subgraph):


        selected_data = [self.graph_list[node] for node in node_subgraph]
    
        loader = Batch.from_data_list(selected_data)

        return loader  
    
   # 3821993,