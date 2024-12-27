#!/usr/bin/env python
# coding: utf-8



import torch
from protein_bert_pytorch import ProteinBERT, PretrainingWrapper
import pandas as pd
import ast
from torch.utils.data import random_split, DataLoader, TensorDataset 
import os,shutil
import pandas as pd  
import numpy as np  
import torch  
from Bio import SeqIO
import argparse 




model = ProteinBERT(
    num_tokens = 21,
    num_annotation = 27,
    dim = 1024,
    dim_global = 256,
    depth = 4,
    narrow_conv_kernel = 9,
    wide_conv_kernel = 9,
    wide_conv_dilation = 5,
    attn_heads = 6,
    attn_dim_head = 36,
    local_to_global_attn = False,
    local_self_attn = True,
    num_global_tokens = 2,
    glu_conv = False
)

class SequenceDataProcessor:  
    def __init__(self, train_df):  
        self.train_df = train_df  
        self.max_length =  1024
        self.vector_size = 27  
        self.code_dict = 'ACDEFGHIKLMNPQRSTVWY'  
        self.process_data()  
  
    def process_data(self):  
        self.train_df["Domain"] = self.train_df["Domain"].apply(ast.literal_eval)  
        self.train_df['Length'] = self.train_df['Sequence'].apply(len)  
        self.train_df = self.train_df[self.train_df['Length'] <= self.max_length]  
        self.train_df = self.train_df.drop(columns='Length')  
        self.train_df["Domain"] = self.train_df["Domain"].apply(self.create_encoding_vector)  
        self.seqs, self.masks, self.annotations = self.encode_sequences() 
  
    def create_encoding_vector(self, positions):  
        encoding_vector = [0] * self.vector_size  
        for pos in positions:  
            if 0 <= pos <= self.vector_size:  
                encoding_vector[pos-1] = 1
        return encoding_vector  
  
    def one_hot_encode(self, sequence):  
        encoding = np.zeros(self.max_length)  
        mask = np.zeros(self.max_length)  
        for i, aa in enumerate(sequence):  
            if aa in self.code_dict and i < self.max_length:  
                encoding[i] = self.code_dict.index(aa)  
                mask[i] = 1  
        return encoding, mask  
  
    def encode_sequences(self):  
        seqs = []  
        masks = []  
        annotations = []  
        for sequence, domain in zip(self.train_df["Sequence"], self.train_df["Domain"]):  
            seq_encoding, seq_mask = self.one_hot_encode(sequence)  
            seqs.append(seq_encoding)  
            masks.append(seq_mask)  
            annotations.append(domain)  
        return np.array(seqs), np.array(masks), np.array(annotations)  
  
    def convert_to_torch_tensors(self):  
    
        self.seqs = torch.from_numpy(np.array([np.array(i).astype(int) for i in self.seqs])).long()  
        self.masks = torch.from_numpy(np.array([np.array(i).astype(bool) for i in self.masks]))  
        self.annotations = torch.from_numpy(np.array([np.array(i).astype(float) for i in self.annotations])).float()  
  
    def get_data(self):  
        return self.seqs, self.masks, self.annotations

def check_positions(vector, positions):  
    for position in positions:    
        if vector[position - 1] == 1:  # 列表索引从0开始，所以要减去1
            return [position,1]  
        else:
            pass
    return [0,0]

def test_model(model, test_loader):  
    total_predict = np.array([])  
    label = np.array([])  
    for batch_seq, batch_annotation, batch_mask in test_loader:  
        # 前向传播  
        predict = model(batch_seq, batch_annotation, mask=batch_mask)  
        binary_predictions = (predict[1] > 0).int()  
          
        if total_predict.size == 0:  
            total_predict = binary_predictions.numpy()  
            label = batch_annotation.numpy()  
        else:  
            # 否则，按行连接total_predict和binary_predictions  
            total_predict = np.concatenate((total_predict, binary_predictions.numpy()), axis=0)  
            label = np.concatenate((label, batch_annotation.numpy()), axis=0)    
    right = []  
    for i in range(len(total_predict)):  
        pos=check_positions(total_predict[i], [1, 2, 3, 5, 27])[0]
        if pos == 1:
            right.append("AFP III")
        elif pos == 2:
            right.append("AFP INSECT")
        elif pos == 3:
            right.append("AFP II")
        elif pos == 5:
            right.append("AFP I")
        elif pos == 27:
            right.append("AFP ASSOCIATE")
        else:
            right.append("NON AFP")
    return right

parser = argparse.ArgumentParser(description='Protein fasta file')
parser.add_argument('--f', type=str, required=True, help='fasta file path')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
args = parser.parse_args()
fasta_file = args.f
model_path = args.model_path
sequences_dict = {}

for record in SeqIO.parse(fasta_file, "fasta"):
    sequences_dict[record.id] = str(record.seq)

test_df = pd.DataFrame(sequences_dict.items(), columns=['NAME', 'Sequence'])
test_df["Domain"]='[]'
test_df['Length'] = test_df['Sequence'].apply(len)
test_df= test_df[test_df['Length'] <= 1024]

processor_test=SequenceDataProcessor(test_df)
processor_test.convert_to_torch_tensors()
seqs_test, masks_test, annotations_test = processor_test.get_data()
test_dataset = TensorDataset(seqs_test, annotations_test, masks_test) 
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


state_dict = torch.load(model_path)  

model.load_state_dict(state_dict)
accuracy = test_model(model, test_loader)
test_df["anno"]=accuracy 
test_df.to_csv(f'../result1/{fasta_file}.predict.csv',sep='\t')





