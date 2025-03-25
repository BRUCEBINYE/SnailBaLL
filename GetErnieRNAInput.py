
import os
import time
import numpy as np
import pandas as pd
import torch


import argparse

#file_path = "./data/idp_test_COX1_pair.txt"

# load input sequence
def load_seq(file_path):
    df = pd.read_csv(file_path, sep=',')
    #df = df.sample(frac=1, random_state=10).reset_index(drop=True)
    seq1_seq = df['seq1']
    seq2_seq = df['seq2']
    lbs = df['label']
    return seq1_seq, seq2_seq, lbs


def df2enrie(file_path, save_path, seq1_out_txt, seq2_out_txt, seq1_out_fasta, seq2_out_fasta, lbs_out_path):
    seq1_seq, seq2_seq, lbs = load_seq(file_path)
    names = range(lbs.shape[0])

    # Sequences 1
    seq1_fasta = pd.DataFrame(data=np.repeat('strings', 2*len(seq1_seq)), dtype = str)
    for i in range(len(seq1_seq)):
        # Output files for ENRIE-RNA embedding extraction: seq1_seq
        seq1_seq[i] = seq1_seq[i].replace("T", "U")
        # Output files for ENRIE-RNA ss prediction: seq1_fasta
        if i==0:
            seq1_fasta.iloc[i] = ">SEQ1_" + str(names[i])
            seq1_fasta.iloc[i+1] = seq1_seq[i]
        else:
            seq1_fasta.iloc[2*i] = ">SEQ1_" + str(names[i])
            seq1_fasta.iloc[2*i+1] = seq1_seq[i]
    seq1_seq.to_csv(save_path + seq1_out_txt, index = False, header = False)
    seq1_fasta.to_csv(save_path + seq1_out_fasta, index = False, header = False)

    # Sequences 2
    seq2_fasta = pd.DataFrame(data=np.repeat('strings', 2*len(seq2_seq)), dtype = str)
    for i in range(len(seq2_seq)):
        # Output files for ENRIE-RNA embedding extraction: seq2_seq
        seq2_seq[i] = seq2_seq[i].replace("T", "U")
        # Output files for ENRIE-RNA ss prediction: seq2_fasta
        if i==0:
            seq2_fasta.iloc[i] = ">SEQ2_" + str(names[i])
            seq2_fasta.iloc[i+1] = seq2_seq[i]
        else:
            seq2_fasta.iloc[2*i] = ">SEQ2_" + str(names[i])
            seq2_fasta.iloc[2*i+1] = seq2_seq[i]
    seq2_seq.to_csv(save_path + seq2_out_txt, index = False, header = False)
    seq2_fasta.to_csv(save_path + seq2_out_fasta, index = False, header = False)

    # Labels
    lbs = torch.tensor(lbs, dtype=torch.long)
    torch.save(lbs, save_path + lbs_out_path)





if __name__ == "__main__":
    
    start = time.time()
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', dest='file_path', type=str, help='# DNASequence file in dataframe')
    parser.add_argument('--savepath', dest='save_path', default='./results/ERNIE_RNA_input_file/', type=str, help="# The path of files")
    parser.add_argument('--seq1out', dest='seq1_out_txt', type=str, help='# Out file Name of sequence 1')
    parser.add_argument('--seq2out', dest='seq2_out_txt', type=str, help='# Out file Name of sequence 2')
    parser.add_argument('--seq1fasta', dest='seq1_out_fasta', type=str, help='# Out Fasta file Name of sequence 1')
    parser.add_argument('--seq2fasta', dest='seq2_out_fasta', type=str, help='# Out Fasta file Name of sequence 2')
    parser.add_argument('--lbsout', dest='lbs_out_path', type=str, help='# Outfile Name of labels')

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    
    df2enrie(args.file_path, args.save_path, args.seq1_out_txt, args.seq2_out_txt, 
             args.seq1_out_fasta, args.seq2_out_fasta, args.lbs_out_path)



#conda activate py38
#cd /data1/yebin/0.2_Species/snailPan
#python GetErnieRNAInput.py --input ./data/sample_20000posi_20000neg_seq_pair_dataframe.txt \
#  --savepath ./results/ERNIE_RNA_input_file/ \
#  --seq1out sample_20000posi_20000neg_ERNIE-RNA_seq1_input.txt \
#  --seq2out sample_20000posi_20000neg_ERNIE-RNA_seq2_input.txt \
#  --seq1fasta sample_20000posi_20000neg_ERNIE-RNA_seq1_input.fasta \
#  --seq2fasta sample_20000posi_20000neg_ERNIE-RNA_seq2_input.fasta \
#  --lbsout sample_20000posi_20000neg_ERNIE-RNA_labels.pth

#### After make the input sequence data
# cd /data1/yebin/ERNIE-RNA
# conda activate ERNIE-RNA
# python extract_embedding.py --seqs_path='/data1/yebin/0.2_Species/snailPan/results/ERNIE_RNA_input_file/sample_20000posi_20000neg_ERNIE-RNA_seq1_input.txt' --device=3
# python extract_embedding.py --seqs_path='/data1/yebin/0.2_Species/snailPan/results/ERNIE_RNA_input_file/sample_20000posi_20000neg_ERNIE-RNA_seq2_input.txt' --device=3

