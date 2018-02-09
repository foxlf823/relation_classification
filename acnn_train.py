import data_pro as pro
import pyt_acnn as pa
import capsulenet
import nse
import os
import torch

import torch.utils.data as D
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import KFold
import logging
import shutil
import argparse

use_cuda = torch.cuda.is_available()
if use_cuda:
    print("cuda is used!!!")
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    print("cuda is not supported, use cpu")


logger = logging.getLogger()
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('--max_length', '-ml', default=99, type=int, help='max length of sentence.')
parser.add_argument('--pos_emb_sz', '-pes', default=25, type=int, help='position emb size.')
parser.add_argument('--relation_number', '-rn', default=19, type=int, help='relation class number.')
parser.add_argument('--conv_output_sz', '-cos', default=1000, type=int, help='convolution kernel output size')
parser.add_argument('--dropout_keep_prob', '-dkp', default=1, type=float, help='dropout keep prob')
parser.add_argument('--window_sz', '-ws', default=3, type=int, help='word window size')
parser.add_argument('--learning_rate', '-lr', default=0.2, type=float, help='Learning rate for SGD.')
parser.add_argument('--l2_norm', '-l2', default=1e-8, type=float, help='l2 normalization')
parser.add_argument('--momentum', '-mm', default=0.9, type=float, help='sgd momentum')
parser.add_argument('--batch_size', '-bs', default=50, type=int, help='Batch size for training.')
parser.add_argument('--epochs', '-epochs', default=100, type=int, help='Number of epochs to train for.')
parser.add_argument('--print_loss_iter', '-pli', default=50, type=int, help='print the training loss every these iterations')
parser.add_argument('--data_dir', '-data', default='./data', help='Directory containing training and test data')
parser.add_argument('--pretrained_emb_dir', '-preemb', default='./data', help='Directory containing pretrained embedding')
parser.add_argument('--result_dir', '-result', default='./result', help='Directory containing results')
parser.add_argument('--model', '-m', default=0, type=int, help='Select models: 0-ACNN, 1-CapsNet, 2-NSE')

print("List all parameters...")
args = parser.parse_args()
args_dict = vars(args)
for key,value in args_dict.items():
    print(key+": "+str(value))
print()

N = args.max_length # max length of sentence
DP = args.pos_emb_sz # position emb size
NP = args.max_length # position emb number
NR = args.relation_number # relation class number
DC = args.conv_output_sz # convolution kernel output size
KP = args.dropout_keep_prob # dropout keep prob
K = args.window_sz # word window size
LR = args.learning_rate # learning rate
L2_NORM = args.l2_norm
BATCH_SIZE = args.batch_size 
epochs = args.epochs
PRINT_LOSS_ITER = args.print_loss_iter
MODEL = args.model

data_dir = args.data_dir
preemb_dir = args.pretrained_emb_dir
result_dir = args.result_dir


if os.path.exists(result_dir):
    shutil.rmtree(result_dir)  
    os.mkdir(result_dir)  
else:
    os.mkdir(result_dir)  

data = pro.load_data(data_dir+'/train.txt')
t_data = pro.load_data(data_dir+'/test.txt')
print("Load data")
word_dict = pro.build_dict(data[0])
t_word_dict = pro.build_dict(t_data[0])
word_dict.update(t_word_dict)

x, y, e1, e2, dist1, dist2 = pro.vectorize(data, word_dict, N)
y = np.array(y).astype(np.int64)
# np_cat[0] - word:0-122, e1, e2, pos1:0-122, pos2:0-122
np_cat = np.concatenate((x, np.array(e1).reshape(-1, 1), np.array(e2).reshape(-1, 1), np.array(dist1), np.array(dist2)),
                        1)
e_x, e_y, e_e1, e_e2, e_dist1, e_dist2 = pro.vectorize(t_data, word_dict, N)
y = np.array(y).astype(np.int64)
eval_cat = np.concatenate(
    (e_x, np.array(e_e1).reshape(-1, 1), np.array(e_e2).reshape(-1, 1), np.array(e_dist1), np.array(e_dist2)), 1)

embed_file = preemb_dir
embedding = pro.load_embedding_from_glove(embed_file, word_dict)

if MODEL == 0:
    model = pa.myCuda(pa.ACNN(N, embedding, DP, NP, K, NR, DC, KP, use_cuda))
    loss_func = pa.NovelDistanceLoss(NR)
    logger.info('MODEL: Using attention CNN')
elif MODEL == 1:
    model = pa.myCuda(capsulenet.CapsuleNet(N, embedding, DP, NP, K, NR, DC, KP, routings=3))
    loss_func = capsulenet.caps_loss
    logger.info('MODEL: Using capsule CNN')
else:
    model = pa.myCuda(nse.NSE(N, embedding, DP, NP, K, NR, DC, KP))
    loss_func = model.loss
    logger.info('MODEL: Using NSE')
    torch.backends.cudnn.enabled = False

#optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=args.momentum, weight_decay=L2_NORM)  
optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=L2_NORM)

del embedding

best = -1

train = torch.from_numpy(np_cat.astype(np.int64))
y_tensor = torch.LongTensor(y)
train_datasets = D.TensorDataset(data_tensor=train, target_tensor=y_tensor)
train_dataloader = D.DataLoader(train_datasets, BATCH_SIZE, True, num_workers=1)

eval = torch.from_numpy(eval_cat.astype(np.int64))
y_tensor = torch.LongTensor(e_y)
eval_datasets = D.TensorDataset(data_tensor=eval, target_tensor=y_tensor)
eval_dataloader = D.DataLoader(eval_datasets, BATCH_SIZE, False, num_workers=1)

for i in range(epochs):
    model.train()
    acc = 0
    loss = 0
    j = 0
    for (b_x_cat, b_y) in train_dataloader:
        bx, be1, be2, bd1, bd2, by = pa.data_unpack(b_x_cat, b_y, N, NP, model.training)
        
        if MODEL == 0:
            wo, rel_weight = model(bx, be1, be2, bd1, bd2)
            l = loss_func(wo, rel_weight, by)
            if i != 0 and i % 20 == 0:
                acc_, _ = loss_func.prediction(wo, rel_weight, by, NR)
                acc += acc_
        elif MODEL == 1:
            y_pred = model(bx, be1, be2, bd1, bd2)  # forward
            l = loss_func(by, y_pred)  # compute loss
            if i != 0 and i % 20 == 0:
                acc_, predict  = capsulenet.predict(by, y_pred) 
                acc += acc_    
        else:
            y_pred, M_t = model(bx, be1, be2, bd1, bd2)  # forward
            l = loss_func(by, y_pred)  # compute loss
            if i != 0 and i % 20 == 0:
                acc_, predict  = model.predict(by, y_pred) 
                acc += acc_
            
        if j!=0 and j % PRINT_LOSS_ITER == 0:
            logger.debug('epoch: {}, batch: {}, loss {}'.format(i, j, l.data[0]))
        j += 1
        optimizer.zero_grad()
        l.backward()
        
        torch.nn.utils.clip_grad_norm(model.parameters(), 15.0)
        
        optimizer.step()
        loss += l
    
    print('epoch:', i, 'training avg loss:', loss.cpu().data.numpy()[0] / j, 'accuracy:', acc/j)
    
    model.eval()   
    eval_acc = 0
    ti = 0
    predicts = []
    for (b_x_cat, b_y) in eval_dataloader:
        bx, be1, be2, bd1, bd2, by = pa.data_unpack(b_x_cat, b_y, N, NP, model.training)
        if MODEL == 0:
            wo, rel_weight = model(bx, be1, be2, bd1, bd2, False)
            eval_acc_, predict = loss_func.prediction(wo, rel_weight, by, NR)
        elif MODEL == 1:
            y_pred = model(bx, be1, be2, bd1, bd2)  
            eval_acc_, predict  = capsulenet.predict(by, y_pred)  
        else:
            y_pred, M_t = model(bx, be1, be2, bd1, bd2)  
            eval_acc_, predict  = model.predict(by, y_pred)
            
        eval_acc += eval_acc_
        predicts.extend(predict.data.tolist())
        ti += 1
        
    print('epoch:', i, 'test_acc:', eval_acc / ti)
    
    if eval_acc / ti > best:
        print('epoch: {}, exceed best {}'.format(i, best))
        best = eval_acc / ti 
        torch.save(model.state_dict(), result_dir+'/{}_acnn_params.pkl'.format(i))
        pro.outputToSem10rc(predicts, result_dir+'/{}_result.txt'.format(i))


# model.load_state_dict(torch.load('acnn_params.pkl'))
