
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

from datasetsMinus import load_inputEmb

def load_checkpoint(filepath, device):

    checkpoint = torch.load(filepath, map_location=device)

    model = Net(checkpoint['d_model'], checkpoint['ffn_hidden'], checkpoint['n_head'], checkpoint['n_layers'], 
                checkpoint['n_hidden'], checkpoint['drop_prob'], checkpoint['n_input'], device=device)

    model.load_state_dict(checkpoint['state_dict'])
    return model


def plot_roc(name, y_true, y_scores, **kwargs):
    print("{} AUC: {:.6f}".format(name, roc_auc_score(y_true, y_scores)))
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    plt.plot(fpr, tpr)
    plt.title(name + ' ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


def plot_aupr(name, y_true, y_scores, **kwargs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    plt.plot(recall, precision)
    plt.title(name + ' AUPR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

def test_result(test_seq1_dir, test_seq2_dir, test_lbs_dir, seed, model, device, name, outfolder='./', batch=True):
    
    X_test, y_test = load_inputEmb(test_seq1_dir, test_seq2_dir, test_lbs_dir)
    testfeatures = torch.tensor(X_test).float()
    testlabels = torch.tensor(y_test).long()
    testset = Data.TensorDataset(testfeatures, testlabels)

    if batch:
        batch_size = 128
    else:
        batch_size = len(testset)
        
    testloader = Data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        model.eval()
        model = model.to(device)
        y_true = []
        y_pred = []
        y_scores = []

        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            
            output = model(x)
            # 假设输出是 logits，通过 softmax 转换为概率
            probs = torch.softmax(output, dim=1)
            # 提取第二个类别的概率（索引为 1）
            y_score = probs[:, 1]
            # 根据概率阈值（通常为 0.5）进行预测
            y_pred0 = (y_score > 0.5).long().cpu().numpy()

            y_true.extend(y.cpu().numpy())
            y_pred.extend(y_pred0)
            y_scores.extend(y_score.cpu().detach().numpy())

            #y_pred0 = model.predict(x)
            #y_pred.extend(y_pred0.cpu().numpy())
            #y_true.extend(y.cpu().numpy())
            #y_score = output[:, 1]
            #y_scores.extend(y_score.cpu().detach().numpy())
    
    print("############ \n")
    print(name)
    # Confusion Matrix
    matrix = confusion_matrix(y_true, y_pred)
    cm = pd.DataFrame(matrix, index=['True_0', 'True_1'], columns=['Pred_0', 'Pred_1'])
    print('Confusion Matrix: \n', cm)
    # AUC value
    aucv = float(format(roc_auc_score(y_true, y_scores), '.4f'))
    # AUPR value
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    auprv = float(format(auc(recall, precision), '.4f'))

    tp = cm.values[1][1]
    fn = cm.values[1][0]
    tn = cm.values[0][0]
    fp = cm.values[0][1]
    acc = float(format((tp+tn) / (tp+tn+fp+fn), '.4f'))
    sn = float(format(tp / (tp + fn), '.4f'))
    sp = float(format(tn / (tn + fp), '.4f'))
    f1 = float(format(f1_score(y_true, y_pred), '.4f'))
    mcc = float(format((matthews_corrcoef(y_true, y_pred)), '.4f'))
    print(f'AUC: {aucv}\nAUPR: {auprv}\nAcc: {acc}\nSn: {sn}\nSp: {sp}\nf1: {f1}\nMcc: {mcc}')

    metr = {'AUC': aucv, 'AUPR': auprv, 'TP': tp, 'FN': fn, 'TN': tn, 'FP': fp, 'Acc': acc, 'Sn': sn, 'Sp': sp, 'F1': f1, 'Mcc': mcc}
    df_metr = pd.DataFrame([metr])
    df_metr.to_csv(outfolder + name + '_Metrics.csv', index=False)

    data = {'y_true': y_true, 'y_pred': y_pred, 'y_score': y_scores}
    df_dict = pd.DataFrame(data)
    df_dict.to_csv(outfolder + name + '_predictLabel.csv', index=False)
    
    plt.figure(dpi=300)
    plot_roc(name, y_true, y_scores, linestyle='--')
    plt.savefig(outfolder + name + '_auc.png')

    plt.figure(dpi=300)
    plot_aupr(name, y_true, y_scores, linestyle='--')
    plt.savefig(outfolder + name + '_aupr.png')

    return y_true, y_pred, y_scores



def predMeanTest(seq1_dir, seq2_dir, lbs_dir, device, name, 
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

    y_true_1, y_pred_1, y_scores_1 = test_result(seq1_dir, seq2_dir, lbs_dir, seed=42, model=model_best1, device=device, name=name+"_group1", outfolder = outfolder, batch=batch)
    y_true_2, y_pred_2, y_scores_2 = test_result(seq1_dir, seq2_dir, lbs_dir, seed=42, model=model_best2, device=device, name=name+"_group2", outfolder = outfolder, batch=batch)
    y_true_3, y_pred_3, y_scores_3 = test_result(seq1_dir, seq2_dir, lbs_dir, seed=42, model=model_best3, device=device, name=name+"_group3", outfolder = outfolder, batch=batch)
    y_true_4, y_pred_4, y_scores_4 = test_result(seq1_dir, seq2_dir, lbs_dir, seed=42, model=model_best4, device=device, name=name+"_group4", outfolder = outfolder, batch=batch)

    dfscore = {'y_true': y_true_1, 'y_score_1': y_scores_1, 'y_scores_2': y_scores_2, 'y_scores_3': y_scores_3, 'y_scores_4': y_scores_4}
    dfscore = pd.DataFrame(dfscore)
    dfscore['y_score_mean'] = dfscore.iloc[:, 1:5].mean(axis=1)
    dfscore['y_pred_mean'] = (dfscore['y_score_mean'] > 0.5).astype(int)
    dfscore.to_csv(outfolder + name + '_Mean_prediction.csv', index=False)

    #return dfscore


#DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
#seq1_dir = "/data1/yebin/ERNIE-RNA/results/ernie_rna_representations/idp_test_COX1_pair_seq1/cls_embedding.npy"
#seq2_dir = "/data1/yebin/ERNIE-RNA/results/ernie_rna_representations/idp_test_COX1_pair_seq2/cls_embedding.npy"
#lbs_dir = "/data1/yebin/0.2_Species/snailPan/results/ERNIE_RNA_input_file/idp_test_COX1_pair_labels.pth"
#dfscore1 = predMeanTest(seq1_dir, seq2_dir, lbs_dir, device=DEVICE, name="Idp_test_COX1", batch=False)



def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seq1_dir', dest='seq1_dir', type=str, help='# ERNIE-RNA cls embedding path of sequence 1')
    parser.add_argument('--seq2_dir', dest='seq2_dir', type=str, help='# ERNIE-RNA cls embedding path of sequence 2')
    parser.add_argument('--lbs_dir', dest='lbs_dir', type=str, help='# Label; sequence 1 and 2 are the same genus: 1') # default=150000
    parser.add_argument('--device', dest='device', type=int, default=1, help='# cuda rank')
    parser.add_argument('--name', dest='name', type=str, help='# Define the task name')
    parser.add_argument('--outfolder', dest='outfolder', type=str, default="./result_metrics/", help='# Output folder')
    parser.add_argument('--m1', dest='m1', type=str, default="./model_training_groups/group1_checkpoint_6_2_2_1_2.pth", help='# Path of the best model 1')
    parser.add_argument('--m2', dest='m2', type=str, default="./model_training_groups/group2_checkpoint_6_2_2_1_2.pth", help='# Path of the best model 2')
    parser.add_argument('--m3', dest='m3', type=str, default="./model_training_groups/group3_checkpoint_6_2_2_1_2.pth", help='# Path of the best model 3')
    parser.add_argument('--m4', dest='m4', type=str, default="./model_training_groups/group4_checkpoint_6_2_2_1_2.pth", help='# Path of the best model 4')
    parser.add_argument('--batch', dest='batch', default=False, help='# Whether batch the samples')
    args = parser.parse_args()
    
    os.makedirs(args.outfolder, exist_ok=True)

    DEVICE = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    predMeanTest(args.seq1_dir, args.seq2_dir, args.lbs_dir, DEVICE, args.name, 
                 args.outfolder, args.m1, args.m2, args.m3, args.m4,
                 args.batch)



if __name__ == "__main__":
    main()


