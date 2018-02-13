import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import pyt_acnn as pa

class NSE(nn.Module): 
    def __init__(self, max_len, embedding, pos_embed_size,
             pos_embed_num, slide_window, class_num,
             num_filters, keep_prob, embfinetune):
        
        super(NSE, self).__init__()
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
        self.lstm_layers = 1
        
        self.pad_emb = pa.myCuda(Variable(torch.zeros(1, self.dw)))
        
        if embfinetune:
            self.other_emb = nn.Parameter(torch.from_numpy(embedding[1:, :]))
        else:
            self.other_emb = pa.myCuda(Variable(torch.from_numpy(embedding[1:, :])))
        
        self.read_lstm = nn.LSTM(self.dw, self.dw, self.lstm_layers)
        self.compose_l1 = nn.Linear(2*self.dw, 2*self.dw)
        self.write_lstm = nn.LSTM(2*self.dw, self.dw, self.lstm_layers)
        
        self.dropout = nn.Dropout(1-self.keep_prob)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.dw, self.dc),
            nn.ReLU(),
            self.dropout,
            nn.Linear(self.dc, self.nr)
            )
        
        self.criterion = nn.CrossEntropyLoss()
        
    def init_hidden(self, bz, hidden_dim):
        # Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (pa.myCuda(Variable(torch.zeros(self.lstm_layers, bz, hidden_dim))),
                pa.myCuda(Variable(torch.zeros(self.lstm_layers, bz, hidden_dim))))
        
    def read(self, M_t, x_t, batch_size, hidden):
        """
        The NSE read operation: Eq. 1-3 in the paper
        x_t: (bz, dw)
        M_t: (bz, n, dw)
        """

        o_t, hidden = self.read_lstm(self.dropout(x_t).view(1, batch_size, -1), hidden) # (1, bz, dw)
        z_t = F.softmax(torch.matmul(M_t, o_t.permute(1, 2, 0)).permute(0, 2, 1), dim=2) # (bz, 1, len)
        m_t = torch.matmul(z_t, M_t).view(batch_size, -1) # (bz, dw)
        return o_t.view(batch_size, -1), m_t, z_t.view(batch_size, -1), hidden

    def compose(self, o_t, m_t):
        """
        The NSE compose operation: Eq. 4
        o_t: bz, dw
        m_t: bz, dw
        """

        c_t = self.compose_l1(torch.cat([o_t, m_t], dim=1))
        return c_t # (bz, 2*dw)

    def write(self, M_t, c_t, z_t, hidden, batch_size):
        """
        The NSE write operation: Eq. 5 and 6. 
        c_t: bz, 2*dw
        z_t: bz, n
        M_t: (bz, n, dw)
        """

        h_t, hidden = self.write_lstm(self.dropout(c_t).view(1, batch_size, 2*self.dw), hidden) # (1, bz, dw)
        z_t_e_k = z_t.view(batch_size, self.n, 1).expand(-1, -1, self.dw) # (bz, n, dw)
        
        M_t = (1-z_t_e_k) * M_t + h_t.permute(1,0,2).expand(-1,self.n,-1) * z_t_e_k
        
        return M_t, h_t.view(batch_size,-1), hidden
    
    
    def forward(self, x, e1, e2, dist1, dist2):
        
        bz = x.data.size()[0]
        
        x_embedding = torch.cat((self.pad_emb, self.other_emb),0) 
        # (bz, n, dw)
        x_embed = torch.matmul(pa.one_hot2(x.contiguous().view(bz,self.n,1), self.vac_len), x_embedding)

        M_t = x_embed
        
        read_hidden = self.init_hidden(bz, self.dw)
        write_hidden = self.init_hidden(bz, self.dw)

        for l in range(self.n):
            
            # (bz, 1, dw)
            x_t = torch.index_select(x_embed, 1, pa.myCuda(Variable(torch.LongTensor([l])))) 
        
            # o_t: bz, dw    m_t: bz, dw    z_t: bz, n
            o_t, m_t, z_t, read_hidden = self.read(M_t, x_t.view(bz, -1), bz, read_hidden)
            
            # c_t: bz, 2*dw
            c_t = self.compose(o_t, m_t)
            
            #  M_t: (bz, n, dw)    h_t: (bz, dw)
            M_t, h_t, write_hidden = self.write(M_t, c_t, z_t, write_hidden, bz)
            
        
        y = self.mlp(h_t)
        return y, M_t
        
    def loss(self, gold, pred):
        
        cost = self.criterion(pred, gold)
        
        return cost
    
    def predict(self, gold, y_pred):
        
        bz = gold.data.size()[0]
        correct = 0
    
        predict = y_pred.max(1)[1]
    
        correct = predict.eq(gold).cpu().sum().data[0]
    
        return correct / bz, predict, 
            
            