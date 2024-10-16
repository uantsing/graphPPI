import numpy as np
import csv
import os
from tqdm import tqdm
import sys
import math
import argparse
import re
from itertools import combinations
import ipdb
parser = argparse.ArgumentParser(description='make_adj_set')

parser.add_argument('--distance', default=12, type=float,
                    help="distance threshold")
args = parser.parse_args()

def dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx**2 + dy**2 + dz**2)


def read_atoms(file, chain="."):
    pattern = re.compile(chain)

    atoms = []
    for line in file:
        line = line.strip()
        if line.startswith("ATOM"):
            type = line[12:16].strip()
            chain = line[21:22]
            if type == "CA" and re.match(pattern, chain):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                atoms.append((x, y, z))

    return atoms


def compute_contacts(atoms, threshold):
    contacts = []
    for i in range(len(atoms)-2):
        for j in range(i+2, len(atoms)):
            if dist(atoms[i], atoms[j]) < threshold:
                contacts.append((i, j))
    return contacts


def pdb_to_cm(file, threshold, chain="."):
    atoms = read_atoms(file, chain)
    return compute_contacts(atoms, threshold)



 

list_all = []
for line in tqdm(open('../uniprot_id_Order.txt')):

    contacts = pdb_to_cm(open('/home/zhouyuanqing/ppdesign/SurfPPD/temp/structure/{}.pdb'.format(line.strip()), "r"), args.distance)
    list_all.append(contacts)

    
list_all = np.array(list_all, dtype='object') # 14417
np.save('edge_list_12.npy',list_all)

