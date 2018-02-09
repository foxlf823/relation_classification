

import torch
from torch import nn
from torch.autograd import Variable
from capsulelayers import DenseCapsule, PrimaryCapsule
import pyt_acnn as pa


class CapsuleNet(nn.Module):

    def __init__(self, max_len, embedding, pos_embed_size,
             pos_embed_num, slide_window, class_num,
             num_filters, keep_prob, routings):
        super(CapsuleNet, self).__init__()
        
        self.dw = embedding.shape[1]# word emb size
        self.vac_len = embedding.shape[0]
        self.dp = pos_embed_size # position emb size
        self.d = self.dw + 2 * self.dp # word representation size
        self.np = pos_embed_num # position emb number
        self.nr = class_num # relation class number
        self.k = slide_window # convolutional window size
        self.n = max_len # sentence length

        self.routings = routings
        
        self.pad_emb = pa.myCuda(Variable(torch.zeros(1, self.dw)))
        self.other_emb = nn.Parameter(torch.from_numpy(embedding[1:, :]))
#         self.other_emb = pa.myCuda(Variable(torch.from_numpy(embedding[1:, :])))
        self.dist1_embedding = nn.Embedding(self.np, self.dp)
        self.dist2_embedding = self.dist1_embedding

        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(self.k, self.d), stride=1, padding=0)
        last = self.n - self.k + 1
        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        # output channel should consider capsule dim, e.g., 32*8=256
        self.primarycaps = PrimaryCapsule(256, 256, 8, kernel_size=(2*self.k, 1), stride=1, padding=0)
        lastlast = last - 2*self.k + 1
        # Layer 3: Capsule layer. Routing algorithm works here.
        self.digitcaps = DenseCapsule(in_num_caps=32*lastlast, in_dim_caps=8,
                                      out_num_caps=self.nr, out_dim_caps=16, routings=routings)

        self.relu = nn.ReLU()

    def forward(self, x, e1, e2, dist1, dist2):
        
        bz = x.data.size()[0]
        
        x_embedding = torch.cat((self.pad_emb, self.other_emb),0) 
        x_embed = torch.matmul(pa.one_hot2(x.contiguous().view(bz,self.n,1), self.vac_len), x_embedding)

        dist1_embed = self.dist1_embedding(dist1) # (batch, length, postion_dim)
        dist2_embed = self.dist2_embedding(dist2) # (batch, length, postion_dim)
        x_concat = torch.cat((x_embed, dist1_embed, dist2_embed), 2) # (batch, length, word_dim+2*postion_dim)
        x_concat = x_concat.view(bz, 1, self.n, self.d)
        
        y = self.relu(self.conv1(x_concat))
        y = self.primarycaps(y)
        y = self.digitcaps(y)
        y = y.norm(dim=-1)

        return y


def caps_loss(by, y_pred):
    '''
    by: (bz)
    y_pred: (bz, nr)
    '''
    y_true = pa.myCuda(Variable(torch.zeros(y_pred.size()))).scatter_(1, by.view(-1, 1), 1.)

    L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
    L_margin = L.sum(dim=1).mean()

    return L_margin


def predict(by, y_pred):
    bz = by.data.size()[0]
    correct = 0

    predict = y_pred.max(1)[1]

    correct = predict.eq(by).cpu().sum().data[0]

    return correct / bz, predict, 




