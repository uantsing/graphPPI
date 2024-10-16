import os
import time
import math
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
from gnn_data import GNN_DATA
# from gnn_models_sag import GIN_Net2, ppi_model
from gnn_models import ppi_model
from utils import Metrictor_PPI, print_file, Multigraph2Big
import ipdb



parser = argparse.ArgumentParser(description='HIGH-PPI_model_training')

parser.add_argument('--ppi_path', default=None, type=str,
                    help="ppi path")

parser.add_argument('--p_feat_matrix', default=None, type=str,
                    help="protein feature matrix")
parser.add_argument('--p_adj_matrix', default=None, type=str,
                    help="protein adjacency matrix")
parser.add_argument('--split', default=None, type=str,
                    help='split method, random, bfs or dfs')
parser.add_argument('--save_path', default=None, type=str,
                    help="save folder")
parser.add_argument('--epoch_num', default=None, type=int,
                    help='train epoch number')
seed_num = 2
np.random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)

# node_num = 14417
# def multi2big_x(x_ori):
#     x_cat = torch.zeros(1, 7)
#     x_num_index = torch.zeros(node_num)
#     for i in range(node_num):
#         x_now = torch.tensor(x_ori[i])
#         x_num_index[i] = torch.tensor(x_now.size(0))
#         x_cat = torch.cat((x_cat, x_now), 0)
#     return x_cat[1:, :], x_num_index

# def multi2big_batch(x_num_index):
#     num_sum = x_num_index.sum()
#     num_sum = num_sum.int()
#     batch = torch.zeros(num_sum)
#     count = 1
#     for i in range(1,node_num):
#         zj1 = x_num_index[:i]
#         zj11 = zj1.sum()
#         zj11 = zj11.int()
#         zj22 = zj11 + x_num_index[i]
#         zj22 = zj22.int()
#         size1 = x_num_index[i]
#         size1 = size1.int()
#         tc = count * torch.ones(size1)
#         batch[zj11:zj22] = tc
#         test = batch[zj11:zj22]
#         count = count + 1
#     batch = batch.int()
#     return batch

# # def multi2big_edge(edge_ori, num_index):
    
#     edge_cat = torch.zeros(2, 1)
#     edge_num_index = torch.zeros(node_num)
#     for i in range(node_num):
#         edge_index_p = edge_ori[i]
#         edge_index_p = np.asarray(edge_index_p)
#         edge_index_p = torch.tensor(edge_index_p.T)
#         edge_num_index[i] = torch.tensor(edge_index_p.size(1))
#         if i == 0:
#             offset = 0
#         else:
#             zj = torch.tensor(num_index[:i])
#             offset = zj.sum()
#         edge_cat = torch.cat((edge_cat, edge_index_p + offset), 1)
#     return edge_cat[:, 1:], edge_num_index


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def train(p_x_all, p_edge_all, model, graph,  loss_fn,
           optimizer, device, result_file_path,  save_path, 
           batch_size=512, epochs = 500, scheduler=None):
    global_step = 0
    global_best_valid_f1 = 0.0
    global_best_valid_f1_epoch = 0


    for epoch in range(epochs):
        recall_sum = 0.0
        precision_sum = 0.0
        f1_sum = 0.0
        loss_sum = 0.0
        steps = math.ceil(len(graph.train_mask) / batch_size)
        model.train()
        random.shuffle(graph.train_mask)


        for step in range(steps):
            if step == steps -1:
                train_edge_id = graph.train_mask[step * batch_size:]
            else:
                train_edge_id = graph.train_mask[step*batch_size: step * batch_size + batch_size]
            node_id = graph.edge_index[:, train_edge_id]

            p_x_1 = p_x_all[node_id[0]]
            p_x_2 = p_x_all[node_id[1]]
            p_edge_1 = p_edge_all[node_id[0]]
            p_edge_2 = p_edge_all[node_id[1]]
            M2B_1 = Multigraph2Big(p_x_1, p_edge_1)
            M2B_2 = Multigraph2Big(p_x_2, p_edge_2)
            loader_full_1= M2B_1.process(list(range(len(p_x_1))))
            loader_full_2= M2B_2.process(list(range(len(p_x_2))))
            p_x_1, p_edge_1, batch_1 = loader_full_1.x.to(torch.float32).to(device), torch.LongTensor(loader_full_1.edge_index.to(torch.int64)).to(device), loader_full_1.batch.to(torch.int64).to(device)
            p_x_2, p_edge_2, batch_2 = loader_full_2.x.to(torch.float32).to(device), torch.LongTensor(loader_full_2.edge_index.to(torch.int64)).to(device), loader_full_2.batch.to(torch.int64).to(device)

            output = model(batch_1, p_x_1, p_edge_1, batch_2, p_x_2, p_edge_2)
            label = graph.edge_attr[train_edge_id]

            label = label.type(torch.FloatTensor).to(device)
            loss = loss_fn(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m = nn.Sigmoid()
            pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

            metrics = Metrictor_PPI(pre_result.cpu().data, label.cpu().data, m(output).cpu().data)
            metrics.show_result()
            recall_sum += metrics.Recall
            precision_sum += metrics.Precision
            f1_sum += metrics.F1
            loss_sum += loss.item()

            global_step += 1
            print_file("epoch: {}, step: {}, Train: label_loss: {}, precision: {}, recall: {}, f1: {}"
                       .format(epoch, step, loss.item(), metrics.Precision, metrics.Recall, metrics.F1))
            
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict()},
                   os.path.join(save_path, 'gnn_model_train.ckpt'))
        valid_pre_result_list = []
        valid_label_list = []
        true_prob_list = []
        valid_loss_sum = 0.0
        model.eval()
        valid_steps = math.ceil(len(graph.val_mask) / batch_size)
        with torch.no_grad():
            for step in range(valid_steps):
                if step == valid_steps - 1:
                    valid_edge_id = graph.val_mask[step * batch_size:]
                else:
                    valid_edge_id = graph.val_mask[step * batch_size: step * batch_size + batch_size]
                node_id = graph.edge_index[:, valid_edge_id]
                p_x_1 = p_x_all[node_id[0]]
                p_x_2 = p_x_all[node_id[1]]
                p_edge_1 = p_edge_all[node_id[0]]
                p_edge_2 = p_edge_all[node_id[1]]
                M2B_1 = Multigraph2Big(p_x_1, p_edge_1)
                M2B_2 = Multigraph2Big(p_x_2, p_edge_2)
                loader_full_1= M2B_1.process(list(range(len(p_x_1))))
                loader_full_2= M2B_2.process(list(range(len(p_x_2))))
                p_x_1, p_edge_1, batch_1 = loader_full_1.x.to(torch.float32).to(device), torch.LongTensor(loader_full_1.edge_index.to(torch.int64)).to(device), loader_full_1.batch.to(torch.int64).to(device)
                p_x_2, p_edge_2, batch_2 = loader_full_2.x.to(torch.float32).to(device), torch.LongTensor(loader_full_2.edge_index.to(torch.int64)).to(device), loader_full_2.batch.to(torch.int64).to(device)


                output = model(batch_1, p_x_1, p_edge_1, batch_2, p_x_2, p_edge_2)
                label = graph.edge_attr[valid_edge_id]
                label = label.type(torch.FloatTensor).to(device)

                loss = loss_fn(output, label)
                valid_loss_sum += loss.item()

                m = nn.Sigmoid()
                pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

                valid_pre_result_list.append(pre_result.cpu().data)
                valid_label_list.append(label.cpu().data)
                true_prob_list.append(m(output).cpu().data)

        valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
        valid_label_list = torch.cat(valid_label_list, dim=0)
        true_prob_list = torch.cat(true_prob_list, dim = 0)

        metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list, true_prob_list)

        metrics.show_result()

        recall = recall_sum / steps
        precision = precision_sum / steps
        f1 = f1_sum / steps
        loss = loss_sum / steps

        valid_loss = valid_loss_sum / valid_steps

        if scheduler != None:
            scheduler.step(loss)
            print_file("epoch: {}, now learning rate: {}".format(epoch, scheduler.optimizer.param_groups[0]['lr']),
                       save_file_path=result_file_path)

        if global_best_valid_f1 < metrics.F1:
            global_best_valid_f1 = metrics.F1
            global_best_valid_f1_epoch = epoch

            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict()},
                       os.path.join(save_path, 'gnn_model_valid_best.ckpt'))



        print_file(
            "epoch: {}, Training_avg: label_loss: {}, recall: {}, precision: {}, F1: {}, Validation_avg: loss: {}, recall: {}, precision: {}, F1: {}, Best valid_f1: {}, in {} epoch"
                .format(epoch, loss, recall, precision, f1, valid_loss, metrics.Recall, metrics.Precision, metrics.F1,
                        global_best_valid_f1, global_best_valid_f1_epoch), save_file_path=result_file_path)



def main():
    args = parser.parse_args()
    ppi_data = GNN_DATA(ppi_path=args.ppi_path)
    ppi_data.generate_data()
    ppi_data.split_dataset(train_valid_index_path='./train_val_split_data/train_val_split_1.json', random_new=True,
                           mode=args.split)
    graph = ppi_data.data

    graph.train_mask = ppi_data.ppi_split_dict['train_index']
    graph.val_mask = ppi_data.ppi_split_dict['valid_index']

    p_x_all = torch.load(args.p_feat_matrix)
    p_x_all = np.array(p_x_all, dtype='object')
    p_edge_all = np.load(args.p_adj_matrix, allow_pickle=True)



    print("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    # graph.to(device)
    model = ppi_model()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                           verbose=True)
    save_path = args.save_path
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    

    save_path = os.path.join(save_path, "gnn_{}".format('training_seed_1'))
    result_file_path = os.path.join(save_path, "valid_results.txt")

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    
    train(p_x_all, p_edge_all, model, graph,  loss_fn, optimizer, device,
          result_file_path,  save_path,
          batch_size=2048, epochs=args.epoch_num, scheduler=scheduler)

if __name__ == "__main__":
    main()

# 528896 对相互作用
# python model_train.py --ppi_path ppi_data_processed.txt --split random --p_feat_matrix ./protein_info/x_list_7.pt --p_adj_matrix ./protein_info/edge_list_12.npy --save_path ./result_save  --epoch_num 500

#python model_train.py --ppi_path ppi_data_Demo.txt --split random --p_feat_matrix ./protein_info/x_list_7.pt --p_adj_matrix ./protein_info/edge_list_12.npy --save_path ./result_save  --epoch_num 500