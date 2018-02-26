
import torch
import torchvision
import torch.autograd as autograd
import torch.utils.data as Data
import torch.optim as optim
import torch.nn as nn
import argparse
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', '-data', default='/home/fox/MNIST_data', help='Directory containing training and test data')
parser.add_argument('--layers', '-l', default=12, type=int, help='net layers, must be divided by 6.')
args = parser.parse_args()

# configurations
data_dir = args.data_dir
input_dims = 28
n_classes = 10 
batch_size = 128
learning_rate = 1.0e-3
training_iters = 30000
display_step = 100
SEED = 1
layers = args.layers//6

def myCuda(input):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        return input.cuda()
    else:
        return input


if SEED != 0:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = False

# loading the mnist data
# MNIST data's shape is (number,28,28) and value is 0~255
# MNIST label is integer and value is 1-10
train_data = torchvision.datasets.MNIST(root=data_dir, 
                                        train=True, 
                                        transform=torchvision.transforms.ToTensor(),
                                        download=False
                                        )
print(train_data.train_data.size())
test_data = torchvision.datasets.MNIST(root=data_dir, 
                                       train = False
                                       )
print(test_data.test_data.size())
# shape (2000, 28, 28) value in range(0,1)
test_x = myCuda(autograd.Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.0)
test_y = myCuda(test_data.test_labels[:2000])

train_loader = Data.DataLoader(train_data, batch_size, shuffle=False, num_workers=1)

        

class ResNet(nn.Module):
    def __init__(self, l1, l2, l3):
        super(ResNet, self).__init__()
        
        channel = 16
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, channel, 5), # (bz, 16, 24, 24)
                nn.BatchNorm2d(channel),
                nn.ReLU()
            )
        
        # (bz, 32, 12, 12)
        self.res1_layers, channel = self._register_residual(l1, channel) 

        # (bz, 64, 6, 6)
        self.res2_layers, channel = self._register_residual(l2, channel)
  
        # (bz, 128, 3, 3)
        self.res3_layers, channel = self._register_residual(l3, channel)
        
        self.fc = nn.Linear(channel*3*3, 10)
        

    def forward(self, x): # (bz, 1, 28, 28)
        
        y = self.conv1(x) # (bz, 16, 24, 24)
        
        module_in_res = 5 # see _def_residual
        i = 1
        l_c1, l_c1_bn, l_c2, l_c2_bn, shortcut_projection = None, None, None, None, None

        for res in self.res1_layers: # (bz, 32, 12, 12)
            
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
        for res in self.res2_layers: # (bz, 32, 12, 12)
            
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
        for res in self.res3_layers: # (bz, 32, 12, 12)
            
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
            shortcut_projection = nn.Conv2d(in_channels, out_channels, 1, stride1)
        else: # the size of output is equal to that of input
            stride1 = 1
            shortcut_projection = None
                
        l_c1 = nn.Conv2d(in_channels, out_channels, 3, stride1, 1)
        l_c1_bn = nn.BatchNorm2d(out_channels)
        l_c2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        l_c2_bn = nn.BatchNorm2d(out_channels)
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


model = myCuda(ResNet(layers,layers,layers))
print(model)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


for iter in range(training_iters):
    model.train()
    
    for step, (batch_x, batch_y) in enumerate(train_loader): 
        # (128,1, 28, 28) reshape to (128, 28*28, 1)
        batch_x = myCuda(autograd.Variable(batch_x))
        batch_y = myCuda(autograd.Variable(batch_y))
        
        pred = model.forward(batch_x)
        
        cost = criterion(pred, batch_y)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
 
        
        if (step + 1) % display_step == 0:
            print("Iter " + str(iter + 1) + ", Step "+str(step+1)+", Avarage Loss: " + "{:.6f}".format(cost.data[0]))

    # validation performance
    model.eval()
    test_output = model.forward(test_x)
    pred_y = torch.max(test_output, 1)[1].data.squeeze()
    accuracy = sum(pred_y == test_y) / float(test_y.size(0))
    print("========> Validation Accuarcy: {:.6f}".format(accuracy))
         

