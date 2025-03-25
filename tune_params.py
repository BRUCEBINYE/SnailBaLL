#conda activate py38
#cd 

import os
import torch
import torch.nn as nn
import torch.optim as optim
import time

from models import Net
from utils import EarlyStopping, LRScheduler


import argparse
from datasetsMinus import *
from train import *
from sklearn.model_selection import KFold
import argparse

DEVICE = torch.device('cuda:0')
torch.cuda.manual_seed(42)
#





seq1_dir = "/data1/yebin/ERNIE-RNA/results/ernie_rna_representations/Training_Stylommatophora_COI/TrainingDataSet_475000posi_475000nega_seq_pair_dataframe_group1_seq1/cls_embedding.npy"
seq2_dir = "/data1/yebin/ERNIE-RNA/results/ernie_rna_representations/Training_Stylommatophora_COI/TrainingDataSet_475000posi_475000nega_seq_pair_dataframe_group1_seq2/cls_embedding.npy"
lbs_dir = "/data1/yebin/0.2_Species/snailPan/results/ERNIE_RNA_input_file/Training_Stylommatophora_COI/TrainingDataSet_475000posi_475000nega_seq_pair_dataframe_group1_labels.pth"
use_emb, all_y = load_inputEmb(seq1_dir, seq2_dir, lbs_dir)
print(use_emb.shape)
print(all_y.shape)

batch_size=128

datasets = []
for _ in range(2):
    trainloader, valloader = process_data(use_emb, all_y, val_size=0.2, batch_size=batch_size, seed=42)
    datasets.append((trainloader, valloader))

n_input = use_emb.shape[-2:]
lr = 0.001
drop_prob = 0.5
epochs = 100
#
d_model = 6
ffn_hidden = 2
n_head = 2
n_layers = 1
n_hidden = 2


run(datasets, d_model, ffn_hidden, n_head, 
    n_layers, n_hidden, 
    drop_prob, n_input, epochs,
    lr, DEVICE, save_path='./model_Training_group4/', save_model=True)

# group1  # 97.11
# group2  # 97.68
# group3  # 97.57
# group4  # 98.08





