from model import BertClassifier
import torch
from transformers import BertTokenizer


labels = {'business': 0,
          'entertainment':1,
          'sport': 2,
          'tech': 3,
          'politics': 4}

model = BertClassifier()
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
state_dict = torch.load('checkpoint.std')
model.load_state_dict(state_dict=state_dict['state_dict'])
input = tokenizer('Hello my name is Luong', padding='max_length', max_length=512, truncation=True, return_tensors='pt')

output = model(input['input_ids'], input['attention_mask'])
print(list(labels.items())[output.argmax(dim=1)])
