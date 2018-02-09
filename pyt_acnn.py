# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F



def myCuda(input):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        return input.cuda()
    else:
        return input
    
def data_unpack(cat_data, target, N, NP, train):
    list_x = np.split(cat_data.numpy(), [N, N + 1, N + 2, N + 2 + NP], 1)
    bx = myCuda(Variable(torch.from_numpy(list_x[0]), volatile=not train))
    be1 = myCuda(Variable(torch.from_numpy(list_x[1]), volatile=not train))
    be2 = myCuda(Variable(torch.from_numpy(list_x[2]), volatile=not train))
    bd1 = myCuda(Variable(torch.from_numpy(list_x[3]), volatile=not train))
    bd2 = myCuda(Variable(torch.from_numpy(list_x[4]), volatile=not train))
    target = myCuda(Variable(target, volatile=not train))
    return bx, be1, be2, bd1, bd2, target
    
def one_hot1(indices, depth, value=1):

    onehot = myCuda(Variable(torch.FloatTensor(indices.size(0), depth)))
    onehot.zero_()
    onehot.scatter_(1, indices, value)
    return onehot

def one_hot2(indices, depth, value=1):
    onehot = myCuda(Variable(torch.FloatTensor(indices.size(0), indices.size(1), depth)))
    onehot.zero_()
    onehot.scatter_(2, indices, value)
    return onehot

class ACNN(nn.Module): 
    def __init__(self, max_len, embedding, pos_embed_size,
             pos_embed_num, slide_window, class_num,
             num_filters, keep_prob, use_cuda):
        
        super(ACNN, self).__init__()
        self.dw = embedding.shape[1]# word emb size
        self.vac_len = embedding.shape[0]
        self.dp = pos_embed_size # position emb size
        self.d = self.dw + 2 * self.dp # word representation size
        self.np = pos_embed_num # position emb number
        self.nr = class_num # relation class number
        self.dc = num_filters # convolutional filter output size
        self.keep_prob = keep_prob # dropout keep probability
        self.k = slide_window # convolutional window size
        self.p = (self.k - 1) // 2 # see self.pad
        self.n = max_len # sentence length
        self.kd = self.d * self.k # convolutional filter input size
        self.dropout = nn.Dropout(1-self.keep_prob)
        
        self.pad_emb = Variable(torch.zeros(1, self.dw))
        if use_cuda:
            self.pad_emb = self.pad_emb.cuda()
            

        self.other_emb = nn.Parameter(torch.from_numpy(embedding[1:, :]))
        
        self.dist1_embedding = nn.Embedding(self.np, self.dp)
        self.dist2_embedding = self.dist1_embedding
        
        self.softmax = nn.Softmax(dim=1)
        
        self.conv = nn.Conv2d(1, self.dc, (self.k, self.d), (1, self.d), (self.p, 0))  
        self.tanh = nn.Tanh()
        
        self.y_embedding = nn.Parameter(torch.randn(self.nr, self.dc))
        self.U = nn.Parameter(torch.randn(self.dc, self.nr))
        
        self.max_pool = nn.MaxPool2d((1, self.dc), (1, self.dc))
        
        self.residual = nn.Parameter(torch.randn(self.d, self.dc))
        self.max_pool1 = nn.MaxPool2d((self.n, 1), (self.n, 1))
    
    def forward(self, x, e1, e2, dist1, dist2, is_training=True):
        bz = x.data.size()[0]
        
        x_embedding = torch.cat((self.pad_emb, self.other_emb),0) 
        x_embed = torch.matmul(one_hot2(x.contiguous().view(bz,self.n,1), self.vac_len), x_embedding)
        e1_embed = torch.matmul(one_hot2(e1.contiguous().view(bz,1,1), self.vac_len), x_embedding)
        e2_embed = torch.matmul(one_hot2(e2.contiguous().view(bz,1,1), self.vac_len), x_embedding)
        
        dist1_embed = self.dist1_embedding(dist1) # (batch, length, postion_dim)
        dist2_embed = self.dist2_embedding(dist2) # (batch, length, postion_dim)
        x_concat = torch.cat((x_embed, dist1_embed, dist2_embed), 2) # (batch, length, word_dim+2*postion_dim)
        
        A1 = torch.matmul(x_embed, e1_embed.permute(0,2,1)) # (batch, length, 1)
        A2 = torch.matmul(x_embed, e2_embed.permute(0,2,1))
        alpha1 = self.softmax(A1) 
        alpha2 = self.softmax(A2)
        
        alpha = torch.div(torch.add(alpha1, alpha2), 2) 
        
        R = torch.mul(x_concat, alpha) # (batch, length, word_dim+2*postion_dim)
        
        R = self.conv(R.view(bz, 1, self.n, self.d))  # (batch, self.dc, length, 1)
        R = self.tanh(R)  
        R_star = R.view(bz, self.dc, self.n) # (batch, self.dc, length)
        
        rel_weight = self.y_embedding
        temp_U = self.U
        
        G = torch.matmul(R_star.permute(0,2,1), temp_U) # (bz, n, nr)
        G = torch.matmul(G, rel_weight) # (bz, n, dc)

        AP = self.softmax(G)

        wo = torch.bmm(R_star, AP)   # (batch, self.dc, self.dc)
        wo = self.max_pool(wo.view(bz, 1, self.dc, self.dc)) # (batch, 1, self.dc, 1)
        wo = wo.view(bz, self.dc)
        
        resi = torch.matmul(x_concat,self.residual)
        resi = self.max_pool1(resi.view(bz, 1, self.n, self.dc)).view(bz, self.dc)
        
        wo = wo+resi
        
        return wo, rel_weight # (batch, self.dc), (num_relation, self.dc)

    
class NovelDistanceLoss(nn.Module):
    def __init__(self, nr, margin=1):
        super(NovelDistanceLoss, self).__init__()
        self.nr = nr
        self.margin = margin

    def forward(self, wo, rel_weight, in_y): # in_y (bz)
        wo_norm = F.normalize(wo)  # (bz, dc)
        bz = wo_norm.data.size()[0]
        dc = wo_norm.data.size()[1]
        
        rel_weight_norm = F.normalize(rel_weight) # (nr, dc)
        
        # find neg_y
        temp1 = wo_norm.view(bz, 1, dc) - rel_weight_norm # (bz, nr, dc)
        all_distance = torch.norm(temp1, 2, 2)  # (bz, nr)

        mask = one_hot1(in_y.view(bz,1), self.nr, 1000)  # (bz, nr)
        masked_y = torch.add(all_distance, mask)
        neg_y = torch.min(masked_y, dim=1)[1]  # (bz,)
        
        # distance function
        neg_y = torch.mm(one_hot1(neg_y.view(bz,1), self.nr), rel_weight_norm)  # (bz, nr)*(nr, dc) => (bz, dc)
        pos_y = torch.mm(one_hot1(in_y.view(bz,1), self.nr), rel_weight_norm)
        neg_distance = torch.norm(wo_norm - neg_y, 2, 1)
        pos_distance = torch.norm(wo_norm - pos_y, 2, 1)
        
        temp2 = torch.clamp(self.margin - neg_distance, 0, 9999)
        
        # objective
        loss = torch.mean(pos_distance + temp2)
        return loss
    
    # predict relation based on the distance of vectors
    def prediction(self, wo, rel_weight, y, NR):
        wo_norm = F.normalize(wo)
        bz = wo_norm.data.size()[0]
        dc = wo_norm.data.size()[1]
        
        rel_weight_norm = F.normalize(rel_weight) # (nr, dc)
        temp1 = wo_norm.view(bz, 1, dc) - rel_weight_norm # (bz, nr, dc)
        all_distance = torch.norm(temp1, 2, 2)  # (bz, nr)
        
        predict = torch.min(all_distance, 1)[1].long()

        correct = torch.eq(predict, y)

        acc = correct.sum().float() / float(correct.data.size()[0])
        return (acc * 100).cpu().data.numpy()[0], predict


