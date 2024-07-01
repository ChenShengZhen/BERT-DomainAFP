#!/usr/bin/env python
# coding: utf-8


import torch
from protein_bert_pytorch import ProteinBERT, PretrainingWrapper
import pandas as pd
import ast
from torch.utils.data import random_split, DataLoader, TensorDataset 
from torch.utils.tensorboard import SummaryWriter
import os,shutil
import pandas as pd  
import numpy as np  
import torch  
import ast  
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
    random_replace_token_prob = 0.05,    # 替换为随机令牌的令牌百分比，默认为论文中的5%  
    remove_annotation_prob = 0.25,       # 删除注解的百分比，默认为25%  
    add_annotation_prob = 0.01,          # 随机添加注解的概率，默认为1%  
    remove_all_annotations_prob = 0.5,   # 完全删除批处理项目中注解的百分比，默认为50%  
    seq_loss_weight = 1.,                # 序列损失的权重
    annotation_loss_weight = 1.,         # 注解损失的权重  
    exclude_token_ids = ()        # 用于排除填充、开始和结束令牌，使其不被掩盖
)

# do the following in a loop for a lot of sequences and annotations


# In[3]:


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
        if position < 1 or position > len(vector):  
            print(f"位置 {position} 超出了向量的范围")
            return [0,0]  
        elif vector[position - 1] == 1:  # 列表索引从0开始，所以要减去1
            return [position,1]  
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
          
        #print(len(total_predict), len(label))  
    right = 0  
    for i in range(len(total_predict)):  
        if check_positions(total_predict[i], [1, 2, 3, 4, 5, 27])[1] == check_positions(label[i], [1, 2, 3, 4, 5, 27])[1]:  
            right += 1  
    AFP_predict = {0:0,1:0,2:0,3:0,5:0,27:0}  
    AFP_total = {0:0,1:0,2:0,3:0,5:0,27:0}  
    for i in range(len(total_predict)):  
        p=check_positions(total_predict[i], [1, 2, 3, 4, 5, 27])
        l=check_positions(label[i], [1, 2, 3, 4, 5, 27])
        if p[0] == l[0]:  
            AFP_predict[l[0]] += 1
            AFP_total[l[0]] += 1
        else:
            AFP_total[l[0]] += 1
    accuracy = right / len(total_predict)  
    print(f"模型准确率: {accuracy * 100:.4f}%")  
    print(f"分类准确率: {AFP_total,AFP_predict}")
    return accuracy

#小批量随机过采样
def oversample_to_balance(df, target_count=400, target_column='Type'):  
    # 初始化一个空的DataFrame来存储过采样后的数据  
    oversampled_df = pd.DataFrame()  
    # 获取所有唯一的Type值  
    unique_types = df[target_column].unique()  
    # 遍历每个唯一的Type值  
    for type_value in unique_types:  
        # 获取当前Type的所有数据  
        type_df = df[df[target_column] == type_value]  
        # 如果当前Type的数据少于target_count条，则直接添加到过采样后的DataFrame中  
        if len(type_df) < target_count:  
            type_df_upsampled = resample(type_df,   
                                          replace=True,    # 允许重复抽样  
                                          n_samples=target_count, # 抽样到目标数量  
                                          # 可以设置random_state以获得可复现的结果  
                                          #random_state=123
                                          )  # 设置随机种子以获得可复现的结果
            oversampled_df = pd.concat([oversampled_df,type_df_upsampled], ignore_index=True) 
        # 如果当前Type的数据超过target_count条，则随机选择target_count条添加到过采样后的DataFrame中  
        else:  
            # 使用resample函数进行随机抽样  
            type_df_upsampled = resample(type_df,   
                                          replace=False,    # 不重复抽样  
                                          n_samples=target_count, # 抽样到目标数量  
                                          #random_state=123
                                          )     # 设置随机种子以获得可复现的结果  
            # 将抽样后的数据添加到过采样后的Da:taFrame中  
            oversampled_df = pd.concat([oversampled_df,type_df_upsampled], ignore_index=True)  
    # 如果某个Type类别的数据本身就少于target_count条，则直接返回过采样后的DataFrame  
    # 因为这些类别已经包含了所有可用的数据  
    return oversampled_df  





test_df = pd.read_csv("AntiFreezeDomains-Test.csv")
processor_test=SequenceDataProcessor(test_df)
processor_test.convert_to_torch_tensors()
seqs_test, masks_test, annotations_test = processor_test.get_data()
test_dataset = TensorDataset(seqs_test, annotations_test, masks_test) 
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
validation_df = pd.read_csv("AntiFreezeDomains-Validation.csv")
processor_validation=SequenceDataProcessor(validation_df)
processor_validation.convert_to_torch_tensors()
seqs_validation, masks_validation, annotations_validation = processor_validation.get_data()
validation_dataset = TensorDataset(seqs_validation, annotations_validation, masks_validation) 
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)




def load_datasets_and_loaders(train_file_path, batch_size=32):  
    # Read training and testing CSV files  
    train_df = pd.read_csv(train_file_path)  
    AFP_df=train_df[~train_df["Type"].isnull()]
    fake_AFP_df=train_df[train_df["Type"].isnull()]
    #AFP_df=oversample_to_balance(AFP_df)
    fake_AFP_df = fake_AFP_df.sample(n=len(AFP_df))
    train_df=pd.concat([AFP_df,fake_AFP_df], ignore_index=True)
    #print(train_df)
    sequence_rows = train_df[train_df['NAME'].str.startswith('Sequence_')]  
    num_fake_sequences = len(sequence_rows)  
    fake_sequence_rows = train_df[train_df['NAME'].str.startswith('fake_Sequence_')] 
    selected_fake_sequences = fake_sequence_rows.sample(n=num_fake_sequences, replace=False) 
    train_df = pd.concat([sequence_rows, selected_fake_sequences]) 
    processor = SequenceDataProcessor(train_df)  
    #processor.imbalance_data()
    processor.convert_to_torch_tensors()  
    seqs, masks, annotations = processor.get_data()  
    train_dataset = TensorDataset(seqs, annotations, masks)  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  
    return train_loader

train_df = "AntiFreezeDomains-Train.csv"


# In[16]:


# 定义优化器  
train_loader = load_datasets_and_loaders(train_df)
for data, labels,mask in train_loader:
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  
num_epochs = 12
for epoch in range(num_epochs):  
    print(f"epoch:{epoch} start")
    model.train()  
    total_loss = 0.0  
    batch_idx = 0  # 初始化批次索引  
    for batch_seq, batch_annotation, batch_mask in train_loader:  
        # 前向传播  
        loss = learner(batch_seq, batch_annotation, mask=batch_mask)  
        # 反向传播和优化  
        loss.backward()  
        optimizer.step()  
        optimizer.zero_grad()  
        # 累积损失以便输出  
        total_loss += loss.item()  
    #train_loader=load_datasets_and_loaders(train_df)

    model.train()
    print(total_loss)
    model.eval()  # 将模型设置为评估模式  
    val_loss = 0.0  
    val_accuracy = 0.0  
    total_predict = np.array([])  
    label = np.array([])  
    with torch.no_grad():  # 不计算梯度，节省内存和计算资源  
        right={0:0,1:0}
        for val_batch_seq, val_batch_annotation, val_batch_mask in validation_loader:  
            val_outputs = model(val_batch_seq, val_batch_annotation, mask=val_batch_mask)  
            val_loss = learner(val_batch_seq,val_batch_annotation, mask=val_batch_mask)    
            binary_predictions = (val_outputs[1] > 0).int()  
            
        
            if total_predict.size == 0:  
                total_predict = binary_predictions.numpy()  
                label = val_batch_annotation.numpy()   
            else:  
                # 否则，按行连接total_predict和binary_predictions  
                total_predict = np.concatenate((total_predict, binary_predictions.numpy()), axis=0)  
                label = np.concatenate((label, val_batch_annotation.numpy()), axis=0)  
        
        for i in range(len(total_predict)):
            if check_positions(total_predict[i], [1, 2, 3, 4, 5, 27]) == check_positions(label[i], [1, 2, 3, 4, 5, 27]):
                #print(check_positions(total_predict[i], [1, 2, 3, 4, 5, 27])[1])
                print(check_positions(total_predict[i], [1, 2, 3, 4, 5, 27]),check_positions(label[i], [1, 2, 3, 4, 5, 27]))
                right[check_positions(total_predict[i], [1, 2, 3, 4, 5, 27])[1]] += 1  
            # 假设你有一个函数来计算准确度
    print(len(total_predict),right)
    val_loss /= len(total_predict)  
    right_percent=(right[0]+right[1]) / len(total_predict)
    best_acc=0
    if right_percent > best_acc:
        torch.save(model.state_dict(), './improved-protein-bert.pt')
        best_acc=right_percent
    print(f"损失{val_loss},准确率{right_percent}")
#在测试集上进行验证1
    if right_percent >0.9:
        accuracy = test_model(model, test_loader)




