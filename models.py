import torch
from torch import nn
import torch.nn.functional as F

import sys
sys.path.append('..')
from blocks.encoder_layer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob) for _ in range(n_layers)])
    def forward(self, x):
        for layer_module in self.layers:
            x = layer_module(x)
               
        return x
    

#class PostionalEncoding(nn.Module):
#    """
#    compute sinusoid encoding.
#    """
#    def __init__(self, d_model, pos, device):
#        """
#        constructor of sinusoid encoding class
#
#        :param d_model: dimension of model
#        :param pos: Series, position of features
#        :param device: hardware device setting
#        """
#        super(PostionalEncoding, self).__init__()
#        self.d_model = d_model
#        self.pos = torch.tensor(pos.values).float().to(device)
#        self.device = device
#
#    def forward(self, x):
#        _, max_len = x.size()
#        encoding = torch.zeros(max_len, self.d_model, device=self.device)
#        encoding.requires_grad = False
#        pos = self.pos
#        pos = pos.float().unsqueeze(dim=1)
#        _2i = torch.arange(0, self.d_model, step=2, device=self.device).float()
#        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
#        # "step=2" means 'i' multiplied with two (same with 2 * i)
#        encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / self.d_model)))
#        encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / self.d_model)))
#        # compute positional encoding to consider positional information of words
#        return encoding
    

class Net(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, n_layers, #pos, 
                 n_hidden, drop_prob, n_input, n_out=2, device='cuda'):
        super(Net, self).__init__()
        
        self.seq_len, self.input_dim = n_input  # n_input = [130, 1024]

        self.linear0 = nn.Linear(self.input_dim, d_model)
        self.encoder = Encoder(d_model=d_model, ffn_hidden=ffn_hidden,
                           n_head=n_head, n_layers=n_layers, drop_prob=drop_prob)
        #self.pos_encoding = PostionalEncoding(d_model=d_model, pos=pos, device=device)
        self.linear1 = nn.Linear(d_model, n_hidden)
        self.linear2 = nn.Linear(self.seq_len * n_hidden, n_out)
        
    def forward(self, x):
        x = self.linear0(x)  # [batch_size, seq_len, input_dim] -> [batch_size, seq_len, d_model]
        x = self.encoder(x)  # [batch_size, seq_len, d_model]
        x = F.relu(self.linear1(x))  # [batch_size, seq_len, n_hidden]
        x = x.view(x.size(0), -1)  # 展平为 [batch_size, seq_len * n_hidden]
        x = self.linear2(x)  # [batch_size, n_out]
        return x
    
    def predict(self, x):
        out = self.forward(x)
        return out.argmax(dim=1)
