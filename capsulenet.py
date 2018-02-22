

import torch
from torch import nn
from torch.autograd import Variable
from capsulelayers import DenseCapsule, PrimaryCapsule
import pyt_acnn as pa
import math
import torch.nn.functional as F


class CapsuleNet(nn.Module):

    def __init__(self, max_len, embedding, pos_embed_size,
             pos_embed_num, slide_window, class_num,
             num_filters, keep_prob, routings, embfinetune, pad_embfinetune, use_crcnn_loss, 
             include_other):
        super(CapsuleNet, self).__init__()
        
        self.dw = embedding.shape[1]# word emb size
        self.vac_len = embedding.shape[0]+1
        self.dp = pos_embed_size # position emb size
        self.d = self.dw + 2 * self.dp # word representation size
        self.np = pos_embed_num # position emb number
        self.include_other = include_other
        self.other_id = class_num-1 # only used when no other
        if include_other:
            self.nr = class_num # relation class number
        else:
            self.nr = class_num-1 # relation class number
        self.k = slide_window # convolutional window size
        self.n = max_len # sentence length
        self.keep_prob = keep_prob # dropout keep probability
        
        self.routings = routings
        self.conv1_out_channel = 256 #256
        self.primarycap_out_channel = 256#256 # 8*32
        self.primarycap_dim = 8 #8
        self.densecap_input_channel = 32#32
        self.densecap_dim = 16#16
        
        if pad_embfinetune:
            self.pad_emb = pa.myCuda(Variable(torch.randn(1, self.dw), requires_grad=True))
        else:
            self.pad_emb = pa.myCuda(Variable(torch.zeros(1, self.dw)))
        
        if embfinetune:
#             self.other_emb = nn.Parameter(torch.from_numpy(embedding[1:, :]))
            self.other_emb = nn.Parameter(torch.from_numpy(embedding[:, :]))
        else:
#             self.other_emb = pa.myCuda(Variable(torch.from_numpy(embedding[1:, :])))
            self.other_emb = pa.myCuda(Variable(torch.from_numpy(embedding[:, :])))
            
#         self.dropout_word = nn.Dropout(1-self.keep_prob)
        
        if self.dp != 0:
            if pad_embfinetune:
                self.pad_pos_emb = pa.myCuda(Variable(torch.randn(1, self.dp), requires_grad=True))
            else:
                self.pad_pos_emb = pa.myCuda(Variable(torch.zeros(1, self.dp)))
            self.other_pos_emb = nn.Parameter(torch.FloatTensor(self.np-1, self.dp))
            self.other_pos_emb.data.normal_(0, 1)
#             self.dist1_embedding = nn.Embedding(self.np, self.dp)
#             self.dist2_embedding = self.dist1_embedding
#             self.dropout_dist1 = nn.Dropout(1-self.keep_prob)
#             self.dropout_dist2 = nn.Dropout(1-self.keep_prob)

        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = nn.Conv2d(1, self.conv1_out_channel, kernel_size=(self.k, self.d), stride=1, padding=0)
        self.last = self.n - self.k + 1
#         self.W_res1_1 = nn.Parameter(torch.FloatTensor(self.d, 1))
#         self.W_res1_1.data.uniform_(-math.sqrt(6. / (self.d)) , math.sqrt(6. / (self.d)))
#         self.W_res1_2 = nn.Parameter(torch.FloatTensor(self.last, self.n))
#         self.W_res1_2.data.uniform_(-math.sqrt(6. / (self.last+self.n)) , math.sqrt(6. / (self.last+self.n)))
#         self.dropout_conv1 = nn.Dropout(1-self.keep_prob)
        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        # output channel should consider capsule dim, e.g., 32*8=256
        self.primarycaps = PrimaryCapsule(self.conv1_out_channel, self.primarycap_out_channel, self.primarycap_dim, kernel_size=(2*self.k, 1), stride=1, padding=0)
        self.lastlast = self.last - 2*self.k + 1
#         self.W_res2_1 = nn.Parameter(torch.FloatTensor(self.last, 1))
#         self.W_res2_1.data.uniform_(-math.sqrt(6. / (self.last)) , math.sqrt(6. / (self.last)))
#         self.W_res2_2 = nn.Parameter(torch.FloatTensor(self.primarycap_dim, self.primarycap_out_channel))
#         self.W_res2_2.data.uniform_(-math.sqrt(6. / (self.primarycap_dim+self.primarycap_out_channel)) , math.sqrt(6. / (self.primarycap_dim+self.primarycap_out_channel)))
#         self.dropout_primary = nn.Dropout(1-self.keep_prob)
        # Layer 3: Capsule layer. Routing algorithm works here.
        self.digitcaps = DenseCapsule(in_num_caps=self.densecap_input_channel*self.lastlast, in_dim_caps=self.primarycap_dim,
                                      out_num_caps=self.nr, out_dim_caps=self.densecap_dim, routings=routings)
#         self.dropout_dense = nn.Dropout(1-self.keep_prob)
#         self.linear = nn.Linear(self.n*self.d, self.nr)
#         self.W_res3_1 = nn.Parameter(torch.FloatTensor(self.d, 1))
#         self.W_res3_1.data.uniform_(-math.sqrt(6. / (self.d)) , math.sqrt(6. / (self.d)))
#         self.W_res3_2 = nn.Parameter(torch.FloatTensor(self.n, self.densecap_dim))
#         self.W_res3_2.data.uniform_(-math.sqrt(6. / (self.n+self.densecap_dim)) , math.sqrt(6. / (self.n+self.densecap_dim)))
        
        self.relu = nn.ReLU()
        
#         x = pa.myCuda(Variable(torch.LongTensor([[word_dict['has']]*self.n]*2)))
#         x_embedding = torch.cat((self.pad_emb, self.other_emb),0) 
#         x_embed = torch.matmul(pa.one_hot2(x.contiguous().view(2,self.n,1), self.vac_len), x_embedding)
#         pass
#         a = nn.Linear(111,222)
#         pass

        self.use_crcnn_loss = use_crcnn_loss
        if use_crcnn_loss:
            self.W_class = nn.Parameter(torch.FloatTensor(self.nr, self.densecap_dim))
            
            stdv = math.sqrt(6. / (self.nr+self.densecap_dim)) 
            self.W_class.data.uniform_(-stdv, stdv)
        


    def forward(self, x, e1, e2, dist1, dist2):
        
        bz = x.data.size()[0]
        
#         x_embedding = torch.cat((self.pad_emb, self.other_emb),0) 
        x_embedding = torch.cat((self.pad_emb, self.other_emb),0) 
        x_embed = torch.matmul(pa.one_hot2(x.contiguous().view(bz,self.n,1), self.vac_len), x_embedding)
#         x_embed = self.dropout_word(x_embed)

        if self.dp !=0:
#             dist1_embed = self.dist1_embedding(dist1) # (batch, length, postion_dim)
#             dist2_embed = self.dist2_embedding(dist2) # (batch, length, postion_dim)
            pos_embedding = torch.cat((self.other_pos_emb, self.pad_pos_emb),0)
            dist1_embed = torch.matmul(pa.one_hot2(dist1.contiguous().view(bz,self.n,1), self.np), pos_embedding)
            dist2_embed = torch.matmul(pa.one_hot2(dist2.contiguous().view(bz,self.n,1), self.np), pos_embedding)
#             dist1_embed = self.dropout_dist1(dist1_embed)
#             dist2_embed = self.dropout_dist2(dist2_embed)

            x_concat = torch.cat((x_embed, dist1_embed, dist2_embed), 2) # (batch, length, word_dim+2*postion_dim)
        else:
            x_concat = x_embed
            
        # input attention
#         e1_embed = torch.matmul(pa.one_hot2(e1.contiguous().view(bz,1,1), self.vac_len), x_embedding)
#         e2_embed = torch.matmul(pa.one_hot2(e2.contiguous().view(bz,1,1), self.vac_len), x_embedding)
#         A1 = torch.matmul(x_embed, e1_embed.permute(0,2,1)) # (batch, length, 1)
#         A2 = torch.matmul(x_embed, e2_embed.permute(0,2,1))
#         alpha1 = F.softmax(A1, dim=1) 
#         alpha2 = F.softmax(A2, dim=1)
#         alpha = torch.div(torch.add(alpha1, alpha2), 2) 
#         R = torch.mul(x_concat, alpha) # (batch, length, word_dim+2*postion_dim)
        
            
        x_concat = x_concat.view(bz, 1, self.n, self.d)
#         x_concat = R.view(bz, 1, self.n, self.d)
        
#         y_conv1 = self.relu(self.conv1(x_concat))
        y_conv1 = self.relu(self.conv1(x_concat))
#         y = self.dropout_conv1(y)
#         y_res1 = y_conv1 + F.relu(torch.matmul(self.W_res1_2, torch.matmul(x_concat, self.W_res1_1)).expand(-1, self.conv1_out_channel, -1, -1))
        
        y_primary = self.primarycaps(y_conv1)
#         y = self.dropout_primary(y)

#         y_res2 = y_primary+F.relu(torch.matmul(self.W_res2_2, torch.matmul(y_res1.squeeze(-1), self.W_res2_1)).permute(0,2,1).expand(-1, self.densecap_input_channel*self.lastlast, -1))
        
        y = self.digitcaps(y_primary) # [bz, nr, dim_caps]
#         y = self.dropout_dense(y)

#         y = y_digit + F.relu(torch.matmul(torch.matmul(x_concat, self.W_res3_1).squeeze(-1), self.W_res3_2).expand(-1, 19, -1))
        
        if self.use_crcnn_loss:
            
            y = torch.matmul(y.view(bz, self.nr, 1, self.densecap_dim), self.W_class.view(self.nr, self.densecap_dim, 1))
            y = y.view(bz, self.nr)
            
        else:
            y = y.norm(dim=-1)

#         y = self.linear(x_concat.view(bz, -1))

        

        return y
    
    def loss_func(self, by, y_pred):
        if self.use_crcnn_loss:
            loss = self._crcnn_loss(by, y_pred)
        else:
            loss = self._caps_loss(by, y_pred)
            
        return loss


    def _crcnn_loss(self, by, y_pred):
        bz = by.size()[0]
        m_pos = 2.5
        m_neg = 0.5
        r = 2
        
        # y_pred (bz, 18), by may contain 'other' (id=18), which leads out of index
        if self.include_other:
            new_by = by
        else:
            other_mask = pa.myCuda(Variable(torch.LongTensor(by.size())))
            other_mask.fill_(self.other_id)
            other_mask.ne_(by)
            new_by = by*other_mask # mask other to 0, although 0 correspond to a class, we will mask its score later
        
        
        y_true = pa.myCuda(Variable(torch.zeros(y_pred.size()))).scatter_(1, new_by.view(-1, 1), 1.)
    
        s_gold = torch.matmul(y_true.view(bz, 1, self.nr), y_pred.view(bz, self.nr, 1)).view(new_by.size())
        
        left = torch.log(1+torch.exp(r*(m_pos - s_gold)))
        
        if self.include_other == False:
            left.mul_(other_mask.float())
            
        
        mask = pa.one_hot1(new_by.view(bz,1), self.nr, -1000)  # mask gold
        if self.include_other == False:
            aaaa = other_mask.view(bz, 1).expand(-1, self.nr).float()
            mask = mask*aaaa # mask fake 0 (actually other)
        
        masked_y = torch.add(y_pred, mask) 
        s_neg = torch.max(masked_y, dim=1)[0] 
        
        right = torch.log(1+torch.exp(r*(m_neg+s_neg)))
        
        loss = left+right
        
        loss = loss.mean()
        
        return loss
            
            
    
    def _caps_loss(self, by, y_pred):
        '''
        by: (bz)
        y_pred: (bz, nr)
        '''
        bz = by.size()[0]
        m_pos = 0.9
        m_neg = 0.1
        
        if self.include_other:
            new_by = by
        else:
            other_mask = pa.myCuda(Variable(torch.LongTensor(by.size())))
            other_mask.fill_(self.other_id)
            other_mask.ne_(by)
            new_by = by*other_mask
                    
        y_true = pa.myCuda(Variable(torch.zeros(y_pred.size()))).scatter_(1, new_by.view(-1, 1), 1.)
        
        if self.include_other == False:
            aaaa = other_mask.view(bz, 1).expand(-1, self.nr).float()
            y_true.mul_(aaaa.float()) # mask fake 0 (actually other)
        
    
        L = y_true * torch.clamp(m_pos - y_pred, min=0.) ** 2 + \
            0.5 * (1 - y_true) * torch.clamp(y_pred - m_neg, min=0.) ** 2
        L_margin = L.sum(dim=1).mean()
    
        return L_margin

    def predict(self, by, y_pred):
        if self.use_crcnn_loss:
            accuracy, answer = self._crcnn_predict(by, y_pred)
        else:
            accuracy, answer = self._cap_predict(by, y_pred)
            
        return accuracy, answer

    def _cap_predict(self, by, y_pred):
        bz = by.data.size()[0]
        correct = 0
        m_neg = 0.1
    
        if self.include_other: 
            predict = y_pred.max(1)[1]
        else: # this code is only correct when other is the last id
            max_score, temp = y_pred.max(1)
            mask1 = max_score.gt(m_neg)
            not_other_predict = mask1.long()*temp
            
            mask2 = max_score.lt(m_neg)
            other_predict = pa.myCuda(Variable(torch.LongTensor(by.size()))).fill_(self.other_id)
            other_predict = mask2.long()*other_predict
            
            predict = not_other_predict + other_predict
            
    
        correct = predict.eq(by).cpu().sum().data[0]
    
        return correct / bz, predict
    
    def _crcnn_predict(self, by, y_pred):
        bz = by.data.size()[0]
        correct = 0
        
        if self.include_other: 
            predict = y_pred.max(1)[1]
        else: # this code is only correct when other is the last id
            max_score, temp = y_pred.max(1)
            mask1 = max_score.gt(0)
            not_other_predict = mask1.long()*temp
            
            mask2 = max_score.lt(0)
            other_predict = pa.myCuda(Variable(torch.LongTensor(by.size()))).fill_(self.other_id)
            other_predict = mask2.long()*other_predict
            
            predict = not_other_predict + other_predict
            
        
        correct = predict.eq(by).cpu().sum().data[0]
    
        return correct / bz, predict





