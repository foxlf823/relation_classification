
import torch
import torchvision
from torch.autograd import Variable
import torch.utils.data as Data
import torch.optim as optim
import torch.nn as nn
import argparse
import torch.nn.functional as F
import pyt_acnn as pa


        

class ResNet(nn.Module):
    def __init__(self, max_len, embedding, pos_embed_size,
             pos_embed_num, slide_window, class_num,
             num_filters, keep_prob, embfinetune, pad_embfinetune, layers):
        super(ResNet, self).__init__()
        
        self.dw = embedding.shape[1]# word emb size
        self.vac_len = embedding.shape[0]+1
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
        
        self.criterion = nn.CrossEntropyLoss()
        
        if pad_embfinetune:
            self.pad_emb = pa.myCuda(Variable(torch.randn(1, self.dw), requires_grad=True))
        else:
            self.pad_emb = pa.myCuda(Variable(torch.zeros(1, self.dw)))
        
        if embfinetune:
            self.other_emb = nn.Parameter(torch.from_numpy(embedding[:, :]))
        else:
            self.other_emb = pa.myCuda(Variable(torch.from_numpy(embedding[:, :])))

        
        if self.dp != 0:
            if pad_embfinetune:
                self.pad_pos_emb = pa.myCuda(Variable(torch.randn(1, self.dp), requires_grad=True))
            else:
                self.pad_pos_emb = pa.myCuda(Variable(torch.zeros(1, self.dp)))
            self.other_pos_emb = nn.Parameter(torch.FloatTensor(self.np-1, self.dp))
            self.other_pos_emb.data.normal_(0, 1)
        
        channel = 16
        if self.dp !=0:
            self.conv1 = nn.Sequential(
                    nn.Conv2d(1, channel, (4,self.d)), # (bz, 16, 96, 1)
                    nn.BatchNorm2d(channel),
                    nn.ReLU()
                )
        else:
            self.conv1 = nn.Sequential(
                    nn.Conv2d(1, channel, (4,self.dw)), # (bz, 16, 96, 1)
                    nn.BatchNorm2d(channel),
                    nn.ReLU()
                )
        
        # (bz, 32, 48)
        self.res1_layers, channel = self._register_residual(layers[0], channel) 

        # (bz, 64, 24)
        self.res2_layers, channel = self._register_residual(layers[1], channel)
  
        # (bz, 128, 12)
        self.res3_layers, channel = self._register_residual(layers[2], channel)
        
        self.fc = nn.Linear(channel*12, self.nr)
        

    def forward(self, x, e1, e2, dist1, dist2): 
        
        bz = x.data.size()[0]
        
        x_embedding = torch.cat((self.pad_emb, self.other_emb),0) 
        x_embed = torch.matmul(pa.one_hot2(x.contiguous().view(bz,self.n,1), self.vac_len), x_embedding)
        
        if self.dp !=0:
            pos_embedding = torch.cat((self.other_pos_emb, self.pad_pos_emb),0)
            dist1_embed = torch.matmul(pa.one_hot2(dist1.contiguous().view(bz,self.n,1), self.np), pos_embedding)
            dist2_embed = torch.matmul(pa.one_hot2(dist2.contiguous().view(bz,self.n,1), self.np), pos_embedding)
            x_concat = torch.cat((x_embed, dist1_embed, dist2_embed), 2) # (batch, length, word_dim+2*postion_dim)
        else:
            x_concat = x_embed # (bz, len, word_dim)
        
        x_concat = x_concat.view(bz, 1, self.n, -1)
        
        y = self.conv1(x_concat) # (bz, 16, 96, 1)
        y = y.squeeze(dim=-1)
        
        module_in_res = 5 # see _def_residual
        i = 1
        l_c1, l_c1_bn, l_c2, l_c2_bn, shortcut_projection = None, None, None, None, None

        for res in self.res1_layers: # (bz, 32, 48)
            
            if i == 1:
                l_c1 = res
            elif i == 2:
                l_c1_bn = res
            elif i == 3:
                l_c2 = res
            elif i == 4:
                l_c2_bn = res
            elif i == 5:
                shortcut_projection = res
            
            if i%module_in_res==0:
                y = self._do_residual(y, l_c1, l_c1_bn, l_c2, l_c2_bn, shortcut_projection)
                i = 0
                
            i += 1
        
        
        i = 1    
        for res in self.res2_layers: # (bz, 64, 24)
            
            if i == 1:
                l_c1 = res
            elif i == 2:
                l_c1_bn = res
            elif i == 3:
                l_c2 = res
            elif i == 4:
                l_c2_bn = res
            elif i == 5:
                shortcut_projection = res
            
            if i%module_in_res==0:
                y = self._do_residual(y, l_c1, l_c1_bn, l_c2, l_c2_bn, shortcut_projection)
                i = 0
                
            i += 1
        
        i = 1    
        for res in self.res3_layers: # (bz, 128, 12)
            
            if i == 1:
                l_c1 = res
            elif i == 2:
                l_c1_bn = res
            elif i == 3:
                l_c2 = res
            elif i == 4:
                l_c2_bn = res
            elif i == 5:
                shortcut_projection = res
            
            if i%module_in_res==0:
                y = self._do_residual(y, l_c1, l_c1_bn, l_c2, l_c2_bn, shortcut_projection)
                i = 0
                
            i += 1
            
        y = self.fc(y.view(y.size(0),-1))

        return y
    
    def _register_residual(self, layers, channel):
        
        res_layers = nn.ModuleList()
        for k in range(layers):
            if k==0:
                res = self._def_residual(channel, channel*2)
                channel = channel*2
                res_layers.extend(res)
            else:
                res = self._def_residual(channel, channel)
                res_layers.extend(res)
        
        return res_layers, channel     
        
    
    def _def_residual(self, in_channels, out_channels):

        if in_channels != out_channels: # channel double, size half
            stride1 = 2
            shortcut_projection = nn.Conv1d(in_channels, out_channels, 1, stride1)
        else: # the size of output is equal to that of input
            stride1 = 1
            shortcut_projection = None
                
        l_c1 = nn.Conv1d(in_channels, out_channels, 3, stride1, 1)
        l_c1_bn = nn.BatchNorm1d(out_channels)
        l_c2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1)
        l_c2_bn = nn.BatchNorm1d(out_channels)
        return [l_c1, l_c1_bn, l_c2, l_c2_bn, shortcut_projection]
    
    def _do_residual(self, input, l_c1, l_c1_bn, l_c2, l_c2_bn, shortcut_projection):
    
        c1 = l_c1(input)
        c1 = l_c1_bn(c1)
        c1 = F.relu(c1)    
        
        c2 = l_c2(c1)
        
        if shortcut_projection:
            input = shortcut_projection(input)
        
        c2 = c2 + input
        c2 = l_c2_bn(c2)
        c2 = F.relu(c2)
        
        return c2
    
    def loss(self, gold, pred):
        
        cost = self.criterion(pred, gold)
        
        return cost

    def predict(self, gold, y_pred):
        
        bz = gold.data.size()[0]
        correct = 0
    
        predict = y_pred.max(1)[1]
    
        correct = predict.eq(gold).cpu().sum().data[0]
    
        return correct / bz, predict, 


