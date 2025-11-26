import torch
import torch.nn as nn

class MPRL(nn.Module):
    def __init__(self, num_classes, input_dim =1024, num_filters =64, filter_sizes= [2, 3, 4, 5], hidden_size = 64, num_layers = 2, dropout=0.2, num_heads=4):
        super(MPRL, self).__init__()
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.filter_sizes = [f for f in filter_sizes if f <= input_dim]
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (filter_size, input_dim), padding=(filter_size // 2, 0)) for filter_size in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(input_size=1024,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True,
                              bidirectional=True)
        self.attention1 = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=num_heads, batch_first=True)
        self.attention2 = nn.MultiheadAttention(embed_dim=256, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(256+128, num_classes)

    def forward(self, x ,mask):
        x1 = x.unsqueeze(1) 
        x1 = [conv(x1).squeeze(3) for conv in self.convs] 
        x1 = [torch.relu(item) for item in x1]  
        min_length = min([item.size(2) for item in x1])
        x1 = [item[:, :, :min_length] for item in x1]  
        x1 = torch.cat(x1, 1)  
        x1 = x1.permute(0, 2, 1)  
        x1 = self.dropout(x1) 
        x1,_= self.attention2(x1,x1,x1)

        x, _ = self.bilstm(x)  
        x, attention_weights = self.attention1(x, x, x)
        x = torch.cat((x1,x),dim=2)
        attention_weights = attention_weights.mean(dim=1)  

        x = torch.mean(x, dim=1) 
        out = self.fc(x) 


        return out
