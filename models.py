import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# https://www.kaggle.com/gennadylaptev/factorization-machine-implemented-in-pytorch
class FM(nn.Module):
    
    def __init__(self, n=None, k=None):
        super().__init__()
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients
        self.V = nn.Parameter(torch.randn(n, k),requires_grad=True)
        self.lin = nn.Linear(n, 1)

        
    def forward(self, x):
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True) #S_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True) # S_2
        
        out_inter = 0.5*(out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin
        batch_size = out.shape[0]
        out = torch.sigmoid(out.view(batch_size))
        
        return out
    
    
    def predict(self, x):
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True) #S_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True) # S_2
        
        out_inter = 0.5*(out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin
        batch_size = out.shape[0]
        out = torch.sigmoid(out.view(batch_size))
        
        return torch.mean(out)
    
    
# 推薦にしか使えない
# つまり特徴量のうち非ゼロ要素(1)が２つ
class NFM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, user_size, item_size):
        super(nfm, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.user_embed = nn.Embedding(user_size, embedding_dim)
        self.item_embed = nn.Embedding(item_size, embedding_dim)
        
        self.lin0 = nn.Linear(embedding_dim, hidden_dim)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, 1)
        
        self.drop = nn.Dropout()
        
        self.user_bias = nn.Embedding(user_size, 1)
        self.item_bias = nn.Embedding(item_size, 1)
        
        self.global_bias = nn.Embedding(1, 1)
        self.global_bias_id = torch.tensor(0, device=device, dtype=torch.long)
        
    def forward(self, user_tensor, item_tensor):
        # user, itemをembed
        user_embed = self.user_embed(user_tensor)
        item_embed = self.item_embed(item_tensor)
        
        interaction_embed = self.drop(user_embed * item_embed)
        
        
        pred = self.lin3(F.relu(self.lin1(F.relu(self.lin0(interaction_embed)))))
        
        pred = self.user_bias(user_tensor) + self.item_bias(item_tensor)+ self.global_bias(self.global_bias_id)
        return F.sigmoid(pred)
    
    #def init_hidden(self):
        # hiddenレイヤーの初期化

    

class BPR(nn.Module):

    def __init__(self, embedding_dim, user_size, item_size):
        super(BPR, self).__init__()
        self.embedding_dim = embedding_dim

        self.user_embed = nn.Embedding(user_size, embedding_dim)
        self.item_embed = nn.Embedding(item_size, embedding_dim)
        
        
    def forward(self, user_tensor, item_tensor, nega_item_tensor):
        # user, itemをembed
        user_embed = self.user_embed(user_tensor)
        item_embed = self.item_embed(item_tensor)
        nega_item_embed = self.item_embed(nega_item_tensor)
        
        
        interaction_embed = torch.sum(user_embed * item_embed, 1)
        nega_interaction_embed = torch.sum(user_embed * nega_item_embed, 1)
        
        prob = torch.sigmoid(interaction_embed - nega_interaction_embed)
        
        return prob
    
    def predict(self, user_tensor, item_tensor):
        user_embed = self.user_embed(user_tensor)
        item_embed = self.item_embed(item_tensor)
        interaction_embed = torch.sum(user_embed * item_embed, 1)
        
        mu = torch.mean(interaction_embed)
        prob = torch.sigmoid(interaction_embed)
        
        return prob
        