#!/usr/bin/env python
# coding: utf-8

import torch
from protein_bert_pytorch import ProteinBERT, PretrainingWrapper
import pandas as pd
import ast
from torch.utils.data import random_split, DataLoader, TensorDataset 
from torch.utils.tensorboard import SummaryWriter
import os, shutil
import numpy as np  
from sklearn.utils import resample

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

learner = PretrainingWrapper(
    model,
    random_replace_token_prob = 0.05,  
    remove_annotation_prob = 0.25,       
    add_annotation_prob = 0.01,          
    remove_all_annotations_prob = 0.5,   
    seq_loss_weight = 1.,                
    annotation_loss_weight = 1.,         
    exclude_token_ids = ()        
)

class SequenceDataProcessor:  
    def __init__(self, train_df):  
        self.train_df = train_df  
        self.max_length = 1024
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
        if position < 1 or position > len(vector):  
            print(f"Position {position} is out of range")
            return [0, 0]  
        elif vector[position - 1] == 1:  
            return [position, 1]  
    return [0, 0]

def test_model(model, test_loader):  
    total_predict = np.array([])  
    label = np.array([])  
    for batch_seq, batch_annotation, batch_mask in test_loader:  
        predict = model(batch_seq, batch_annotation, mask=batch_mask)  
        binary_predictions = (predict[1] > 0).int()  
          
        if total_predict.size == 0:  
            total_predict = binary_predictions.numpy()  
            label = batch_annotation.numpy()  
        else:  
            total_predict = np.concatenate((total_predict, binary_predictions.numpy()), axis=0)  
            label = np.concatenate((label, batch_annotation.numpy()), axis=0)  

    right = 0  
    for i in range(len(total_predict)):  
        if check_positions(total_predict[i], [1, 2, 3, 4, 5, 27])[1] == check_positions(label[i], [1, 2, 3, 4, 5, 27])[1]:  
            right += 1  
    AFP_predict = {0: 0, 1: 0, 2: 0, 3: 0, 5: 0, 27: 0}  
    AFP_total = {0: 0, 1: 0, 2: 0, 3: 0, 5: 0, 27: 0}  
    for i in range(len(total_predict)):  
        p = check_positions(total_predict[i], [1, 2, 3, 4, 5, 27])  
        l = check_positions(label[i], [1, 2, 3, 4, 5, 27])  
        if p[0] == l[0]:  
            AFP_predict[l[0]] += 1
            AFP_total[l[0]] += 1
        else:
            AFP_total[l[0]] += 1
    accuracy = right / len(total_predict)  
    print(f"Model Accuracy: {accuracy * 100:.2f}%")  
    print(f"Classification Accuracy: {AFP_total, AFP_predict}")
    return accuracy

def oversample_to_balance(df, target_count=150, target_column='Type'):  
    oversampled_df = pd.DataFrame()  
    unique_types = df[target_column].unique()  
    for type_value in unique_types:  
        type_df = df[df[target_column] == type_value]  
        if len(type_df) < target_count:  
            type_df_upsampled = resample(type_df, replace=True, n_samples=target_count)  
            oversampled_df = pd.concat([oversampled_df, type_df_upsampled], ignore_index=True) 
        else:  
            type_df_upsampled = resample(type_df, replace=False, n_samples=target_count)  
            oversampled_df = pd.concat([oversampled_df, type_df_upsampled], ignore_index=True)  
    return oversampled_df  

test_df = pd.read_csv("AntiFreezeDomains-Test.csv")
processor_test = SequenceDataProcessor(test_df)
processor_test.convert_to_torch_tensors()
seqs_test, masks_test, annotations_test = processor_test.get_data()
test_dataset = TensorDataset(seqs_test, annotations_test, masks_test) 
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

validation_df = pd.read_csv("AntiFreezeDomains-Validation.csv")
processor_validation = SequenceDataProcessor(validation_df)
processor_validation.convert_to_torch_tensors()
seqs_validation, masks_validation, annotations_validation = processor_validation.get_data()
validation_dataset = TensorDataset(seqs_validation, annotations_validation, masks_validation) 
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

def load_datasets_and_loaders(train_file_path, batch_size=32):  
    train_df = pd.read_csv(train_file_path)  
    AFP_df = train_df[~train_df["Type"].isnull()]
    fake_AFP_df = train_df[train_df["Type"].isnull()]
    AFP_df = oversample_to_balance(AFP_df)
    fake_AFP_df = fake_AFP_df.sample(n=len(AFP_df))
    train_df = pd.concat([AFP_df, fake_AFP_df], ignore_index=True)
    sequence_rows = train_df[train_df['NAME'].str.startswith('Sequence_')]  
    num_fake_sequences = len(sequence_rows)  
    fake_sequence_rows = train_df[train_df['NAME'].str.startswith('fake_Sequence_')] 
    selected_fake_sequences = fake_sequence_rows.sample(n=num_fake_sequences, replace=False) 
    train_df = pd.concat([sequence_rows, selected_fake_sequences]) 
    processor = SequenceDataProcessor(train_df)  
    processor.convert_to_torch_tensors()  
    seqs, masks, annotations = processor.get_data()  
    train_dataset = TensorDataset(seqs, annotations, masks)  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  
    return train_loader

train_df = "AntiFreezeDomains-Train.csv"

train_loader = load_datasets_and_loaders(train_df)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  
num_epochs = 12
for epoch in range(num_epochs):  
    print(f"epoch: {epoch} start")
    model.train()  
    total_loss = 0.0  
    for batch_seq, batch_annotation, batch_mask in train_loader:  
        loss = learner(batch_seq, batch_annotation, mask=batch_mask)  
        loss.backward()  
        optimizer.step()  
        optimizer.zero_grad()  
        total_loss += loss.item()  

    model.eval()  
    val_loss = 0.0  
    val_accuracy = 0.0  
    total_predict = np.array([])  
    label = np.array([])  
    with torch.no_grad():  
        right = 0
        for val_batch_seq, val_batch_annotation, val_batch_mask in validation_loader:  
            val_outputs = model(val_batch_seq, val_batch_annotation, mask=val_batch_mask)  
            val_loss = learner(val_batch_seq, val_batch_annotation, mask=val_batch_mask)    
            binary_predictions = (val_outputs[1] > 0).int()  
            
            if total_predict.size == 0:  
                total_predict = binary_predictions.numpy()  
                label = val_batch_annotation.numpy()   
            else:  
                total_predict = np.concatenate((total_predict, binary_predictions.numpy()), axis=0)  
                label = np.concatenate((label, val_batch_annotation.numpy()), axis=0)  

    for i in range(len(total_predict)):
        if check_positions(total_predict[i], [1, 2, 3, 4, 5, 27]) == check_positions(label[i], [1, 2, 3, 4, 5, 27]):
            right += 1  
    val_loss /= len(total_predict)  
    right /= len(total_predict)
    best_acc = 0
    if right > best_acc:
        torch.save(model.state_dict(), 'BERT-DomainAFP.pt')
        best_acc = right
    print(f"Loss: {val_loss}, Accuracy: {right}")

accuracy = test_model(model, test_loader)
