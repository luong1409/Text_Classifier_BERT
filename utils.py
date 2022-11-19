import pandas as pd
import numpy as np
from transformers import BertTokenizer
from dataset import Dataset
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

def preprocess(datapath='bbc-text.csv'):
    
    df = pd.read_csv(datapath)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    np.random.seed(112)
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), 
                                    [int(.8*len(df)), int(.9*len(df))])
    
    return df_train, df_val, df_test

def train(model:nn.Module, train_data, val_data, learning_rate, epochs):
    
    train, val = Dataset(train_data), Dataset(val_data)
    
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=learning_rate)
    
    
    model = model.to(device=device)
    criterion = criterion.to(device=device)
    
    
    for epoch_num in range(epochs):
        
        total_acc_train = 0
        total_loss_train = 0
        
        for train_input, train_label in tqdm(train_dataloader):
            
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            
            output = model(input_id, mask)
            
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
        total_acc_val = 0
        total_loss_val = 0
        
        with torch.no_grad():
            
            for val_input, val_label in val_dataloader:
                
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
                
                output = model(input_id, mask)
                
                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1) == val_label).sum()
                total_acc_val += acc
                
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
            | Train Accuracy: {total_acc_train / len(train_data): .3f} \
            | Val Loss: {total_loss_val / len(val_data): .3f} \
            | Val Accuracy: {total_acc_val / len(val_data): .3f}')
