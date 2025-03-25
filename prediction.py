
import os
import time
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import argparse

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import matthews_corrcoef, f1_score

import torch
import torch.utils.data as Data

from models import Net

from datasetsMinus import load_inputEmb_new


def load_checkpoint(filepath, device):

    checkpoint = torch.load(filepath, map_location=device)

    model = Net(checkpoint['d_model'], checkpoint['ffn_hidden'], checkpoint['n_head'], checkpoint['n_layers'], 
                checkpoint['n_hidden'], checkpoint['drop_prob'], checkpoint['n_input'], device=device)

    model.load_state_dict(checkpoint['state_dict'])
    return model



def pred_result(new_seq1_dir, new_seq2_dir, seed, model, device, name, outfolder='./', batch=False):

    # Prediction sequence pairs with no known label

    X_test = load_inputEmb_new(new_seq1_dir, new_seq2_dir)
    testfeatures = torch.tensor(X_test).float()
    #testlabels = torch.tensor(y_test).long()
    testset = Data.TensorDataset(testfeatures)

    if batch:
        batch_size = 128
    else:
        batch_size = len(testset)
    testloader = Data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        model.eval()
        model = model.to(device)
        y_pred = []
        y_scores = []

        for x, in testloader:
            x = x.to(device)
            output = model(x)
            probs = torch.softmax(output, dim=1)
            y_score = probs[:, 1]
            y_pred0 = (y_score > 0.5).long().cpu().numpy()
            y_pred.extend(y_pred0)
            y_scores.extend(y_score.cpu().detach().numpy())

    print("############ \n")
    print(name)

    data = {'y_pred': y_pred, 'y_score': y_scores}
    df_dict = pd.DataFrame(data)
    df_dict.to_csv(outfolder + name + '_predictLabel.csv', index=False)

    return y_pred, y_scores



def predMeanNew(seq1_dir, seq2_dir, device, name, 
             outfolder = "./result_metrics/", 
             m1 = "./model_training_groups/group1_checkpoint_6_2_2_1_2.pth",
             m2 = "./model_training_groups/group2_checkpoint_6_2_2_1_2.pth",
             m3 = "./model_training_groups/group3_checkpoint_6_2_2_1_2.pth",
             m4 = "./model_training_groups/group4_checkpoint_6_2_2_1_2.pth",
             batch = False):
    model_best1 = load_checkpoint(m1, device)
    model_best2 = load_checkpoint(m2, device)
    model_best3 = load_checkpoint(m3, device)
    model_best4 = load_checkpoint(m4, device)

    y_pred_1, y_scores_1 = pred_result(seq1_dir, seq2_dir, seed=42, model=model_best1, device=device, name=name+"_group1", outfolder = outfolder, batch=batch)
    y_pred_2, y_scores_2 = pred_result(seq1_dir, seq2_dir, seed=42, model=model_best2, device=device, name=name+"_group2", outfolder = outfolder, batch=batch)
    y_pred_3, y_scores_3 = pred_result(seq1_dir, seq2_dir, seed=42, model=model_best3, device=device, name=name+"_group3", outfolder = outfolder, batch=batch)
    y_pred_4, y_scores_4 = pred_result(seq1_dir, seq2_dir, seed=42, model=model_best4, device=device, name=name+"_group4", outfolder = outfolder, batch=batch)

    dfscore = {'y_score_1': y_scores_1, 'y_scores_2': y_scores_2, 'y_scores_3': y_scores_3, 'y_scores_4': y_scores_4}
    dfscore = pd.DataFrame(dfscore)
    dfscore['y_score_mean'] = dfscore.iloc[:, :4].mean(axis=1)
    dfscore['y_pred_mean'] = (dfscore['y_score_mean'] > 0.5).astype(int)
    dfscore.to_csv(outfolder + name + '_Mean_prediction.csv', index=False)

    return dfscore


#DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
#seq1_dir = "/data1/yebin/ERNIE-RNA/results/ernie_rna_representations/idp_test_COX1_pair_seq1/cls_embedding.npy"
#seq2_dir = "/data1/yebin/ERNIE-RNA/results/ernie_rna_representations/idp_test_COX1_pair_seq2/cls_embedding.npy"
#lbs_dir = "/data1/yebin/0.2_Species/snailPan/results/ERNIE_RNA_input_file/idp_test_COX1_pair_labels.pth"
#dfscore1 = predMeanTest(seq1_dir, seq2_dir, lbs_dir, device=DEVICE, name="Idp_test_COX1", batch=False)



def main():
    start = time.time()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seq1_dir', dest='seq1_dir', type=str, help='# ERNIE-RNA cls embedding path of sequence 1')
    parser.add_argument('--seq2_dir', dest='seq2_dir', type=str, help='# ERNIE-RNA cls embedding path of sequence 2')
    parser.add_argument('--device', dest='device', type=int, default=1, help='# cuda rank')
    parser.add_argument('--name', dest='name', type=str, help='# Define the task name')
    parser.add_argument('--outfolder', dest='outfolder', type=str, default="./result_metrics_NoLabel/", help='# Output folder')
    parser.add_argument('--m1', dest='m1', type=str, default="./model_training_groups/group1_checkpoint_6_2_2_1_2.pth", help='# Path of the best model 1')
    parser.add_argument('--m2', dest='m2', type=str, default="./model_training_groups/group2_checkpoint_6_2_2_1_2.pth", help='# Path of the best model 2')
    parser.add_argument('--m3', dest='m3', type=str, default="./model_training_groups/group3_checkpoint_6_2_2_1_2.pth", help='# Path of the best model 3')
    parser.add_argument('--m4', dest='m4', type=str, default="./model_training_groups/group4_checkpoint_6_2_2_1_2.pth", help='# Path of the best model 4')
    parser.add_argument('--batch', dest='batch', default=False, help='# Whether batch the samples')
    args = parser.parse_args()
    
    os.makedirs(args.outfolder, exist_ok=True)

    DEVICE = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    predMeanNew(args.seq1_dir, args.seq2_dir, DEVICE, args.name, 
                 args.outfolder, args.m1, args.m2, args.m3, args.m4,
                 args.batch)

    print(f'Done in {time.time()-start}s!')

if __name__ == "__main__":
    main()
