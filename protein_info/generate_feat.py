import numpy as np
import csv
import os
from tqdm import tqdm
import sys
import math
import argparse
import re
import torch
from itertools import combinations

all_for_assign = np.loadtxt("all_assign.txt")
def read_atoms(file, chain="."):
    pattern = re.compile(chain)
    ajs = []
    for line in file:
        line = line.strip()
        if line.startswith("ATOM"):
            type = line[12:16].strip()
            chain = line[21:22]
            if type == "CA" and re.match(pattern, chain):
                ajs_id = line[17:20]
                ajs.append(ajs_id)
    return ajs

def pdb_to_x(file, chain="."):
    xx = read_atoms(file, chain)
    x_p = np.zeros((len(xx), 7))
    for j in range(len(xx)):
        if xx[j] == 'ALA':
            x_p[j] = all_for_assign[0,:]
        elif xx[j] == 'CYS':
            x_p[j] = all_for_assign[1,:]
        elif xx[j] == 'ASP':
            x_p[j] = all_for_assign[2,:]
        elif xx[j] == 'GLU':
            x_p[j] = all_for_assign[3,:]
        elif xx[j] == 'PHE':
            x_p[j] = all_for_assign[4,:]
        elif xx[j] == 'GLY':
            x_p[j] = all_for_assign[5,:]
        elif xx[j] == 'HIS':
            x_p[j] = all_for_assign[6,:]
        elif xx[j] == 'ILE':
            x_p[j] = all_for_assign[7,:]
        elif xx[j] == 'LYS':
            x_p[j] = all_for_assign[8,:]
        elif xx[j] == 'LEU':
            x_p[j] = all_for_assign[9,:]
        elif xx[j] == 'MET':
            x_p[j] = all_for_assign[10,:]
        elif xx[j] == 'ASN':
            x_p[j] = all_for_assign[11,:]
        elif xx[j] == 'PRO':
            x_p[j] = all_for_assign[12,:]
        elif xx[j] == 'GLN':
            x_p[j] = all_for_assign[13,:]
        elif xx[j] == 'ARG':
            x_p[j] = all_for_assign[14,:]
        elif xx[j] == 'SER':
            x_p[j] = all_for_assign[15,:]
        elif xx[j] == 'THR':
            x_p[j] = all_for_assign[16,:]
        elif xx[j] == 'VAL':
            x_p[j] = all_for_assign[17,:]
        elif xx[j] == 'TRP':
            x_p[j] = all_for_assign[18,:]
        elif xx[j] == 'TYR':
            x_p[j] = all_for_assign[19,:]
    
    return x_p

list_all = []

for line in tqdm(open('../uniprot_id_Order.txt')):
    x = pdb_to_x(open('/home/zhouyuanqing/ppdesign/SurfPPD/temp/structure/{}.pdb'.format(line.strip()), "r"))
    list_all.append(x)
torch.save(list_all, 'x_list_7.pt')

