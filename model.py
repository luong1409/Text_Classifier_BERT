from torch import nn
from transformers import BertModel

class BertClassifier(nn.Module):
    
    def __init__(self, dropout=0.5):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path='bert-base-cased')
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_features=768, out_features=5)
        self.relu = nn.ReLU()
        
        
    def forward(self, input_id, mask):
        
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        '''
        The first variable **_** contains embedding vectors of all the tokens in a sequence.
        The second variable **pooled_output** contains the embedding vector of [CLS] token.
        '''
        dropout_output = self.dropout(pooled_output)
        linear_ouput = self.linear(dropout_output)
        final_output = self.relu(linear_ouput)
        
        return final_output