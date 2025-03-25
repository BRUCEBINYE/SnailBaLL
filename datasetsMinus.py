
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.utils import class_weight
import torch
import torch.utils.data as Data
import tensorflow as tf



def getSequenceData(emb_dir):
    if emb_dir.endswith('.pth'):
        emb = torch.load(emb_dir)
    elif emb_dir.endswith('.npy'):
        emb = torch.from_numpy(np.load(emb_dir)).float()
    else:
        raise ValueError(f"Unsupported file format: {emb_dir}")
    return emb


def getLabel(label_dir):
    if label_dir.endswith('.pth'):
        label = torch.load(label_dir)
    elif label_dir.endswith('.npy'):
        label = torch.from_numpy(np.load(label_dir)).float()
    else:
        raise ValueError(f"Unsupported file format: {label_dir}")
    return label



def load_inputEmb(seq1_dir, seq2_dir, lbs_dir):
    seq1_emb = getSequenceData(seq1_dir).numpy()
    seq2_emb = getSequenceData(seq2_dir).numpy()
    all_y = getLabel(lbs_dir).numpy()
    use_emb = seq2_emb - seq1_emb
    return use_emb, all_y


def load_inputEmb_new(seq1_dir, seq2_dir):
    seq1_emb = getSequenceData(seq1_dir).numpy()
    seq2_emb = getSequenceData(seq2_dir).numpy()
    use_emb = seq2_emb - seq1_emb
    return use_emb


#def dense1(emb):
#    use_emb = tf.keras.layers.Dense(256)(emb)
#    return use_emb



def process_data(rawdata, label, val_size, batch_size, seed):

    # split data
    X_train, X_val, y_train, y_val = train_test_split(rawdata, label, test_size=val_size, stratify=label, random_state=seed)

    # convert the data into tensor
    trainfeatures = torch.tensor(X_train).float()
    trainlabels = torch.tensor(y_train).long()
    valfeatures = torch.tensor(X_val).float()
    vallabels = torch.tensor(y_val).long()

    torch.manual_seed(seed)

    # load the training data
    trainset = Data.TensorDataset(trainfeatures, trainlabels)
    trainloader = Data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    # load the val data
    valset = Data.TensorDataset(valfeatures, vallabels)
    valloader = Data.DataLoader(dataset=valset, batch_size=batch_size, shuffle=True)
    
    return trainloader, valloader



def process_data_k(dat_train, lb_train, dat_val, lb_val, batch_size, seed):
    # convert the data into tensor
    trainfeatures = torch.tensor(dat_train).float()
    trainlabels = torch.tensor(lb_train).long()
    valfeatures = torch.tensor(dat_val).float()
    vallabels = torch.tensor(lb_val).long()

    torch.manual_seed(seed)

    # load the training data
    trainset = Data.TensorDataset(trainfeatures, trainlabels)
    trainloader = Data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    # load the val data
    valset = Data.TensorDataset(valfeatures, vallabels)
    valloader = Data.DataLoader(dataset=valset, batch_size=batch_size, shuffle=True)
    
    return trainloader, valloader

