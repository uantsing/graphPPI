import os
import json
import numpy as np
import copy
import torch
import random

from tqdm import tqdm

from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader

class GNN_DATA:
    def __init__(self, ppi_path, skip_head=True, p1_index=0, p2_index=1, 
                  label_index = 2):
        self.ppi_list = []
        self.ppi_dict = {}
        self.ppi_label_list = []
        self.protein_dict = {}
        self.protein_name = {}
        self.ppi_path = ppi_path

        name = 0
        ppi_name = 0
        self.node_num = 0
        self.edge_num = 0

        class_map = {'reaction':0,
                      'compound':1, 
                      'binding':2, 
                      'activation':3, 
                      'expression':4, 
                      'inhibition':5, 
                      'ubiquitination':6, 
                      'dephosphorylation':7, 
                      'phosphorylation':8, 
                      'repression':9, 
                      'ptmod':10, 
                      'state change':11, 
                      'catalysis':12}
        
        for line in tqdm(open(ppi_path)):
            if skip_head:
                skip_head = False
                continue
            line = line.strip().split('\t')
            # get node and node name
            if line[p1_index] not in self.protein_name.keys():
                self.protein_name[line[p1_index]] = name
                name += 1

            if line[p2_index] not in self.protein_name.keys():
                self.protein_name[line[p2_index]] = name
                name += 1
            
            # get edge and its label
            temp_data = ""
            zj1 = line[p1_index]
            zj2 = line[p2_index]
            if line[p1_index] < line[p2_index]:
                temp_data = line[p1_index] + "__" + line[p2_index]
            else:
                temp_data = line[p2_index] + "__" + line[p1_index]
            if temp_data not in self.ppi_dict.keys():
                self.ppi_dict[temp_data] = ppi_name
                temp_label = [0] * 13
                temp_label[class_map[line[label_index]]] = 1
                self.ppi_label_list.append(temp_label)
                ppi_name += 1
            else:
                index = self.ppi_dict[temp_data]
                temp_label = self.ppi_label_list[index]
                temp_label[class_map[line[label_index]]] = 1
                self.ppi_label_list[index] = temp_label

        i = 0
        for ppi in tqdm(self.ppi_dict.keys()):
            name = self.ppi_dict[ppi]
            assert name == i
            i += 1
            temp = ppi.strip().split('__')
            self.ppi_list.append(temp)
           
        ppi_num = len(self.ppi_list)
        self.origin_ppi_list = copy.deepcopy(self.ppi_list)
        assert len(self.ppi_list) == len(self.ppi_label_list)
        for i in tqdm(range(ppi_num)):
            seq1_name = self.ppi_list[i][0]
            seq2_name = self.ppi_list[i][1]
            # print(len(self.protein_name))
            self.ppi_list[i][0] = self.protein_name[seq1_name]
            self.ppi_list[i][1] = self.protein_name[seq2_name]

        
        self.node_num = len(self.protein_name)
        self.edge_num = len(self.ppi_list)

    def generate_data(self):
        ppi_list = np.array(self.ppi_list)
        ppi_label_list = np.array(self.ppi_label_list)

        self.edge_index = torch.tensor(ppi_list, dtype=torch.long)
        self.edge_attr = torch.tensor(ppi_label_list, dtype=torch.long)
        self.data = Data(x=None, edge_index=self.edge_index.T, edge_attr = self.edge_attr)

    def split_dataset(self, train_valid_index_path, test_size=0.2, random_new=False, mode='random'):
        
        if random_new:
            if mode == 'random':
                ppi_num = self.edge_num
                random_list = [i for i in range(ppi_num)]
                random.shuffle(random_list)

                self.ppi_split_dict = {}
                self.ppi_split_dict['train_index'] = random_list[:int(ppi_num * (1 - test_size))]
                self.ppi_split_dict['valid_index'] = random_list[int(ppi_num * (1 - test_size)):]
                jsobj = json.dumps(self.ppi_split_dict)
                with open(train_valid_index_path, 'w') as f:
                    f.write(jsobj)
                    f.close()

        else:
            with open(train_valid_index_path, encoding='utf-8-sig',errors='ignore') as f:
                str = f.read()
                self.ppi_split_dict = json.loads(str, strict=False)
                f.close()

        


