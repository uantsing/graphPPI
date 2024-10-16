import pandas as pd
from Bio import SeqIO
import ipdb
import mygene
import json


##########################################Begin##############################################################
# //get all symbol

df = pd.read_csv('ppi_data.txt', sep='\t', header=None)
protein_symbol = []
for i, row in df.iterrows():
    if row[0] not in protein_symbol:
        protein_symbol.append(row[0])

    if row[1] not in protein_symbol:
        protein_symbol.append(row[1])

print('number of protein symbol: {}'.format(len(protein_symbol)))
f = open('protein_symbol.txt', 'w')
for name in protein_symbol:
    f.write(name+'\n')

###########################################End#############################################################  


##########################################Begin##############################################################
#//convert symbol to uniprot id

mg = mygene.MyGeneInfo() 
with open('protein_symbol.txt', 'r') as f:
    symbol = f.readlines()
    symbol = [i.strip() for i in symbol]

out = mg.querymany(symbol, scopes='symbol', fields='uniprot', species='human')

symbol2uniprot = {}
for item in out:
    if 'uniprot' in list(item.keys()) and 'Swiss-Prot' in list(item['uniprot'].keys()):
        symbol2uniprot[item['query']] = item['uniprot']['Swiss-Prot']
with open ('symbol2uniprot.json', 'w') as f:
    json.dump(symbol2uniprot, f)

with open('symbol2uniprot.json', 'r') as f:
    loaded_data = json.load(f)

f = open('uniprotid.txt', 'w')

for i in list(loaded_data.values()):
    try:
        f.write(i+'\n')
    except:
        f.write(i[0]+'\n')
f.close()


###########################################End############################################################# 

##########################################Begin##############################################################
#//get pdb file of proteins from alphafold protein structure database

# run in terminal : for i in `cat uniprotid.txt`; do wget -q -N -O ./structure/${i}.pdb https://alphafold.ebi.ac.uk/files/AF-${i}-F1-model_v1.pdb;done

import os
f = open('uniprotid_order.txt', 'w')
for name in os.listdir('./structure'):
    f.write(name.split('.')[0]+'\n')
f.close()

###########################################End############################################################# 

##########################################Begin############################################################## 

with open('symbol2uniprot.json', 'r') as f:
    loaded_data = json.load(f)

for key, value in loaded_data.items():
   if isinstance(value, list):
      loaded_data[key] = value[0]

with open ('symbol2_to_uniprot.json', 'w') as f:
    json.dump(loaded_data, f)
   

PPI = []
df = pd.read_csv('ppi_data.txt', sep='\t', header=None)

for i, row in df.iterrows():
    try:
      PPI.append((loaded_data[row[0]], loaded_data[row[1]], row[2]))
    except:
       print(row)

P1 = []
P2 = []
I = []
for p1, p2, i in PPI:
   P1.append(p1)
   P2.append(p2)
   I.append(i)

print("PPI types: {}".format(set(I)))

dict_ppi = {'p1': P1,
            'p2': P2,
            'interaction': I}

df = pd.DataFrame(dict_ppi)
   
df.to_csv('ppi_data_processed.txt', sep ='\t', index=False)

###########################################End############################################################# 


##########################################Begin############################################################## 

from tqdm import tqdm

ppi_path = "ppi_data_processed.txt"
df = pd.read_csv(ppi_path, sep='\t')
cut = int(df.shape[0] * 0.1)
df[:cut].to_csv("ppi_data_demo.txt", sep="\t", index=False)

p1_index = 0
p2_index = 1

protein_name = {}
name = 0
skip_head = True
for line in tqdm(open("ppi_data_Demo.txt")):
    if skip_head:
        skip_head = False
        continue
    line = line.strip().split('\t')
    # get node and node name
    if line[p1_index] not in protein_name.keys():
        protein_name[line[p1_index]] = name
        name += 1

    if line[p2_index] not in protein_name.keys():
        protein_name[line[p2_index]] = name
        name += 1
    

node_num = len(protein_name)

print('node_num: {}'.format(node_num)) 

f = open('uniprot_id_Order.txt', 'w')

i = 0
for key, value in protein_name.items():
    assert i == value
    f.write(key+'\n')
    i += 1
# 51553 条ppi, 7823 nodes
###########################################End############################################################# 


#########################################Begin############################################################## 

# number = pd.read_csv("error_p_num.txt", header=None)[0].to_list()
# protein = pd.read_csv("uniprot_id_order.txt", header=None)[0]
# for i, p in enumerate(protein):
#     if i in number:
#         protein.drop(index=i, inplace=True)

# print(protein.shape) # 7839
# protein.to_csv('uniprot_id_Order.txt', sep='\t', index=False)

# df = pd.read_csv('ppi_data_demo.txt', sep='\t')
# print(df.shape)
# for i, row in df.iterrows(): 
#     if row['p1'] not in protein.to_list() or row['p2'] not in protein.to_list():
#         df.drop(index=i, inplace=True)

# print(df.shape)
# df.to_csv('ppi_data_Demo.txt', sep='\t', index=False) # 51553
  
###########################################End############################################################# 

