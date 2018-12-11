import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torchvision import datasets, transforms
from resnet import * 

import os
import pickle as pk
from argparse import ArgumentParser
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import multiprocessing as mp
import time
from sklearn.metrics import *
import random
print(torch.cuda.get_device_name(0))

def dump(data, filename):
    with open(filename, 'wb') as f:
        pk.dump(data, f)

def load(filename):
    with open(filename, 'rb') as f:
        data = pk.load(f)
    return data

def gen_todo_list(directory, check = None):
    files = os.listdir(directory)
    todo_list = []
    for f in files:
        fullpath = os.path.join(directory, f)
        if os.path.isfile(fullpath):
            if check is not None:
                if check(f):
                    todo_list.append(fullpath)
            else:
                todo_list.append(fullpath)
    return todo_list

def check(filename):
    return not '_class' in filename
    

def load_data():
    max_data_nb = 10000
    directory = '400x50'
    todo_list = gen_todo_list(directory, check = check)
    ### ver 1 ###
    train_rate = 0.8
    val_rate = 0.18
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []

    for counter, filename in enumerate(todo_list):
        print(filename.split('.')[:-1])
        (tmpX, tmpy) = load(filename)
        
        
        tmpy = load('.'.join(filename.split('.')[:-1]) + '_class.pickle')
        tmpX , tmpy = tmpX[:max_data_nb], tmpy[:max_data_nb]
        assert(len(tmpX) == len(tmpy))
        tmpX ,tmpy= processX(tmpX,tmpy)
        
        random.shuffle(tmpX)
        
        train_num = int(len(tmpX) * train_rate)
        val_num = int(len(tmpX) * val_rate)
        X_train.extend(tmpX[:train_num])
        y_train.extend(tmpy[:train_num])
        X_val.extend(tmpX[train_num: train_num + val_num])
        y_val.extend(tmpy[train_num: train_num + val_num])
        X_test.extend(tmpX[train_num + val_num:])
        y_test.extend(tmpy[train_num + val_num:])
        print('\rLoading... {}/{}'.format(counter+1,len(todo_list)), end = '')
    print('\r{} Data loaded.               '.format(len(todo_list)))
    return X_train, y_train, X_val, y_val, X_test, y_test
    """
    ### ver2 ###
    cpus = mp.cpu_count() - 2
    oldtime = time.time()
    pool = mp.Pool(processes=cpus)
    manager = mp.Manager()
    ns = manager.Namespace()
    ns.X_train = []
    ns.y_train = []
    ns.X_val = []
    ns.y_val = []
    ns.X_test = []
    ns.y_test = []

    res = pool.map(task, [(ns, i, len(todo_list)) for i in todo_list])
    pool.close()
    pool.join()
    #pool.apply_async(task, (ns, i,))
    newtime = time.time()
    print('Using time:', newtime - oldtime, '(sec)')
    return ns.X_train, ns.y_train, ns.X_val, ns.y_val, ns.X_test, ns.y_test
    """


def processX(X,Y):
    if True:
        '''
        X = np.array(X)
        lens = [len(x) for x in X] 
        maxlen = 1500
        tmpX = np.zeros((len(X), maxlen))
        mask = np.arange(maxlen) < np.array(lens)[:,None]
        tmpX[mask] = np.concatenate(X)
        return tmpX
        '''
        cnt =0
        tmpY = []
        for i in range(len(X)):
            if(len(X[i]) > 2 ):
                cnt+=1
        tmpX = np.zeros((cnt , 51 , 400))
        cnt =0
        for i in range(len(X)):
            if(len(X[i]) > 2):
                for j in range(len(X[i])):
                    for k in range(len(X[i][j])):
                        tmpX[cnt][j][k] = X[i][j][k]
                tmpY.append(Y[i])
                cnt+=1
        #print(tmpX.shape)
        return tmpX , tmpY
    else:
        for i, x in enumerate(X):
            tmp_x = np.zeros((1500,))
            tmp_x[:len(x)] = x
            X[i] = tmp_x
        return X

    
'''class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        node_num = 16
        self.conv10 = nn.Conv2d(1, node_num, kernel_size=(3,5) , padding = (1,2))
        #self.conv1_drop = nn.Dropout2d(p=0.05)
        self.conv10_bn = nn.BatchNorm2d(node_num)
        self.conv11 = nn.Conv2d(node_num, node_num, kernel_size=(3,3), padding = 1)
        self.conv11_bn = nn.BatchNorm2d(node_num)        
        self.conv12 = nn.Conv2d(node_num, node_num, kernel_size=(3,3), padding = 1)
        self.conv12_bn = nn.BatchNorm2d(node_num)        
        self.conv13 = nn.Conv2d(node_num, node_num, kernel_size=(3,3), padding = 1)
        self.conv13_bn = nn.BatchNorm2d(node_num)        
        self.conv14 = nn.Conv2d(node_num, node_num, kernel_size=(3,3), padding = 1)
        self.conv14_bn = nn.BatchNorm2d(node_num)        
        self.conv15 = nn.Conv2d(node_num, node_num, kernel_size=(3,3), padding = 1)
        self.conv15_bn = nn.BatchNorm2d(node_num)        
        self.conv16 = nn.Conv2d(node_num, node_num, kernel_size=(3,3), padding = 1)
        self.conv16_bn = nn.BatchNorm2d(node_num)
        
        self.conv20 = nn.Conv2d(node_num ,2*node_num, kernel_size=1, padding = 0)
        self.conv20_bn = nn.BatchNorm2d(2*node_num)   
        self.conv21 = nn.Conv2d(node_num, 2*node_num, kernel_size=(3,3), padding = 1)
        self.conv21_bn = nn.BatchNorm2d(2*node_num)        
        self.conv22 = nn.Conv2d(2*node_num, 2*node_num, kernel_size=(3,3), padding = 1)
        self.conv22_bn = nn.BatchNorm2d(2*node_num)        
        self.conv23 = nn.Conv2d(2*node_num, 2*node_num, kernel_size=(3,3), padding = 1)
        self.conv23_bn = nn.BatchNorm2d(2*node_num)        
        self.conv24 = nn.Conv2d(2*node_num, 2*node_num, kernel_size=(3,3), padding = 1)
        self.conv24_bn = nn.BatchNorm2d(2*node_num)        
        self.conv25 = nn.Conv2d(2*node_num, 2*node_num, kernel_size=(3,3), padding = 1)
        self.conv25_bn = nn.BatchNorm2d(2*node_num)        
        self.conv26 = nn.Conv2d(2*node_num, 2*node_num, kernel_size=(3,3), padding = 1)
        self.conv26_bn = nn.BatchNorm2d(2*node_num)
        self.conv27 = nn.Conv2d(2*node_num, 2*node_num, kernel_size=(3,3), padding = 1)
        self.conv27_bn = nn.BatchNorm2d(2*node_num)        
        self.conv28 = nn.Conv2d(2*node_num, 2*node_num, kernel_size=(3,3), padding = 1)
        self.conv28_bn = nn.BatchNorm2d(2*node_num)
        self.conv29 = nn.Conv2d(2*node_num, 2*node_num, kernel_size=(3,3), padding = 1)
        self.conv29_bn = nn.BatchNorm2d(2*node_num)        
        self.conv2a = nn.Conv2d(2*node_num, 2*node_num, kernel_size=(3,3), padding = 1)
        self.conv2a_bn = nn.BatchNorm2d(2*node_num)
        
        #self.conv30 = nn.Conv2d(2*node_num ,2*node_num, kernel_size=1, padding = 0)
        #self.conv30_bn = nn.BatchNorm2d(2*node_num)   
        self.conv31 = nn.Conv2d(2*node_num ,2*node_num, kernel_size=(3,3), padding = 1)
        self.conv31_bn = nn.BatchNorm2d(2*node_num)        
        self.conv32 = nn.Conv2d(2*node_num, 2*node_num, kernel_size=(3,3), padding = 1)
        self.conv32_bn = nn.BatchNorm2d(2*node_num)        
        self.conv33 = nn.Conv2d(2*node_num, 2*node_num, kernel_size=(3,3), padding = 1)
        self.conv33_bn = nn.BatchNorm2d(2*node_num)        
        self.conv34 = nn.Conv2d(2*node_num, 2*node_num, kernel_size=(3,3), padding = 1)
        self.conv34_bn = nn.BatchNorm2d(2*node_num)        
        self.conv35 = nn.Conv2d(2*node_num, 2*node_num, kernel_size=(3,3), padding = 1)
        self.conv35_bn = nn.BatchNorm2d(2*node_num)        
        self.conv36 = nn.Conv2d(2*node_num, 2*node_num, kernel_size=(3,3), padding = 1)
        self.conv36_bn = nn.BatchNorm2d(2*node_num)
        self.conv37 = nn.Conv2d(2*node_num, 2*node_num, kernel_size=(3,3), padding = 1)
        self.conv37_bn = nn.BatchNorm2d(2*node_num)        
        self.conv38 = nn.Conv2d(2*node_num, 2*node_num, kernel_size=(3,3), padding = 1)
        self.conv38_bn = nn.BatchNorm2d(2*node_num)
        
        self.conv40 = nn.Conv2d(2*node_num ,4*node_num, kernel_size=1, padding = 0,stride = (2,2))
        self.conv40_bn = nn.BatchNorm2d(4*node_num)   
        self.conv41 = nn.Conv2d(2*node_num ,4*node_num, kernel_size=(3,3), padding = 1,stride = (2,2))
        self.conv41_bn = nn.BatchNorm2d(4*node_num)        
        self.conv42 = nn.Conv2d(4*node_num, 4*node_num, kernel_size=(3,3), padding = 1)
        self.conv42_bn = nn.BatchNorm2d(4*node_num)        
        self.conv43 = nn.Conv2d(4*node_num, 4*node_num, kernel_size=(3,3), padding = 1)
        self.conv43_bn = nn.BatchNorm2d(4*node_num)        
        self.conv44 = nn.Conv2d(4*node_num, 4*node_num, kernel_size=(3,3), padding = 1)
        self.conv44_bn = nn.BatchNorm2d(4*node_num)        
        self.conv45 = nn.Conv2d(4*node_num, 4*node_num, kernel_size=(3,3), padding = 1)
        self.conv45_bn = nn.BatchNorm2d(4*node_num)        
        self.conv46 = nn.Conv2d(4*node_num, 2*node_num, kernel_size=(3,3), padding = 1)
        self.conv46_bn = nn.BatchNorm2d(2*node_num)
        
        
        #self.flat_bn = nn.BatchNorm1d(11880)
        
        self.fc0 = nn.Linear(19200, 600)
        self.fc0_bn = nn.BatchNorm1d(600)
        
        
        self.fc1 = nn.Linear(600, 500)
        self.fc1_bn = nn.BatchNorm1d(500)
        
        self.fc2 = nn.Linear(500, 400)
        self.fc2_bn = nn.BatchNorm1d(400)
        
        self.fc3 = nn.Linear(400, 300)
        self.fc3_bn = nn.BatchNorm1d(300)
        self.fc4 = nn.Linear(300, 200)
        self.fc4_bn = nn.BatchNorm1d(200)
        
        self.fc5 = nn.Linear(200, 100)
        self.fc5_bn = nn.BatchNorm1d(100)
        self.fc6 = nn.Linear(100, 50)
        self.fc6_bn = nn.BatchNorm1d(50)
        self.fc7 = nn.Linear(50, 12)

    def forward(self, x):
        
        x =   self.conv10(x) 
        x = F.leaky_relu(self.conv10_bn(x)  )
        res = x
        #x = F.dropout(x, training=self.training,p=0.1)
        x = self.conv11(x)
        x = F.leaky_relu( self.conv11_bn( x ) )
        x = F.dropout2d(x, training=self.training,p=0.1)
        x = self.conv12(x)
        x = x + res
        res = x
        x = F.leaky_relu( self.conv12_bn( x )  )
        #x = F.dropout(x, training=self.training,p=0.1)
        x = self.conv13(x)
        x = F.leaky_relu( self.conv13_bn( x) )
        x = F.dropout2d(x, training=self.training,p=0.1)  
        x = self.conv14(x)
        x = x + res
        res = x
        x = F.leaky_relu( self.conv14_bn( x  ) )
        #x = F.dropout(x, training=self.training,p=0.1)
        x = self.conv15(x)
        x = F.leaky_relu( self.conv15_bn( x) )
        x = F.dropout2d(x, training=self.training,p=0.1)  
        x = self.conv16(x)
        x = x + res
        res = self.conv20_bn(self.conv20(x))
        x = F.leaky_relu( self.conv16_bn( x ))
        #x = F.dropout(x, training=self.training,p=0.1)
        ####################################################
        x =  self.conv21(x)
        x = F.relu(self.conv21_bn( x ) )
        x = F.dropout2d(x, training=self.training,p=0.1)
        x = self.conv22(x)
        x = x + res
        res = x
        x = F.relu( self.conv22_bn( x ) )
        #x = F.dropout(x, training=self.training,p=0.1)
        x = self.conv23(x)
        x = F.relu( self.conv23_bn( x) )
        x = F.dropout2d(x, training=self.training,p=0.1)  
        x = self.conv24(x)
        x = x + res
        res = x
        x = F.relu( self.conv24_bn( x )  )
        #x = F.dropout(x, training=self.training,p=0.1)
        x = self.conv25(x)
        x = F.relu( self.conv25_bn( x) )
        x = F.dropout2d(x, training=self.training,p=0.1)  
        x = self.conv26(x)
        x = x + res
        res = x
        x = F.relu( self.conv26_bn( x )  )
        #x = F.dropout(x, training=self.training,p=0.1)
        x = self.conv27(x)
        x = F.relu( self.conv27_bn( x) )
        x = F.dropout2d(x, training=self.training,p=0.1)  
        x = self.conv28(x)
        x = x + res
        res = x
        x = F.relu( self.conv28_bn( x ) + res )
        #x = F.dropout(x, training=self.training,p=0.1)
        x = self.conv29(x)
        x = F.relu( self.conv29_bn( x) )
        x = F.dropout2d(x, training=self.training,p=0.1)  
        x = self.conv2a(x)
        x = x + res
        res =x
        x = F.relu( self.conv2a_bn( x )  )
        #x = F.dropout(x, training=self.training,p=0.1)
        ###################################################################
        x =  self.conv31(x)
        x = F.leaky_relu(self.conv31_bn( x ) )
        x = F.dropout2d(x, training=self.training,p=0.1)
        x = self.conv32(x)
        x = x + res
        res = x
        x = F.leaky_relu( self.conv32_bn( x ) )
        #x = F.dropout(x, training=self.training,p=0.1)
        x = self.conv33(x)
        x = F.leaky_relu( self.conv33_bn( x) )
        x = F.dropout2d(x, training=self.training,p=0.1)  
        x = self.conv34(x)
        x = x + res
        res = x
        x = F.leaky_relu( self.conv34_bn( x ))
        #x = F.dropout(x, training=self.training,p=0.1)
        x = self.conv35(x)
        x = F.leaky_relu( self.conv35_bn( x) )
        x = F.dropout2d(x, training=self.training,p=0.1)  
        x = self.conv36(x)
        x = x + res
        res = x
        x = F.leaky_relu( self.conv36_bn( x ) +res )
        #x = F.dropout(x, training=self.training,p=0.1)
        x = self.conv37(x)
        x = F.leaky_relu( self.conv37_bn( x) )
        x = F.dropout2d(x, training=self.training,p=0.1)  
        x = self.conv38(x)
        x = x + res
        res =self.conv40_bn( self.conv40(x))
        x = F.leaky_relu( self.conv38_bn( x )  )
        #x = F.dropout(x, training=self.training,p=0.1)
        ########################################################
        x =  self.conv41(x)
        x = F.relu(self.conv41_bn( x ) )
        x = F.dropout2d(x, training=self.training,p=0.1)
        x = self.conv42(x)
        x = x + res
        res = x
        x = F.relu( self.conv42_bn( x ) )
        #x = F.dropout(x, training=self.training,p=0.1)
        x = self.conv43(x)
        x = F.relu( self.conv43_bn( x) )
        x = F.dropout2d(x, training=self.training,p=0.1)  
        x = self.conv44(x)
        x = x + res
        res = x
        x = F.relu( self.conv44_bn( x )  )
        #x = F.dropout(x, training=self.training,p=0.1)
        x = self.conv45(x)
        x = F.relu( self.conv45_bn( x) )
        x = F.dropout2d(x, training=self.training,p=0.1)  
        x = self.conv46(x)
        x = F.max_pool2d(  F.relu( self.conv46_bn( x )  ),2)
          
        
        x = x.view( x.size(0),-1)
       
        x = F.relu(self.fc0_bn( self.fc0(x) ))
        
        
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.dropout(x, training=self.training,p=0.15)
        
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.dropout(x, training=self.training,p=0.15)
        
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = F.dropout(x, training=self.training,p=0.15)
        
        x = F.relu(self.fc4_bn( self.fc4(x) ))
        x = F.dropout(x, training=self.training,p=0.15)
        
        x = self.fc5(x)
        x = F.relu( self.fc5_bn( x) )
        x = F.dropout(x, training=self.training,p=0.15)
        x = self.fc6(x)
        x = F.relu( self.fc6_bn( x) )
        x = F.dropout(x, training=self.training,p=0.15)
        
        
        x = self.fc7( x)
        return F.log_softmax(x, dim=1)'''
    
    
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    #weights = np.array([1,2,1,0.7,1,1,2,1,1,1,1,0.7])
    #weights= torch.from_numpy(weights).float().cuda()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target )
        loss.backward()
        optimizer.step()
        if batch_idx % 80 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()) )

def test( model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    ypred =np.array( [])
    ytrue = np.array( [])
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum' ).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return test_loss , correct / len(test_loader.dataset)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 :
        #m.weight.data.normal_(0.0, 0.02)
        nn.init.kaiming_normal_(m.weight.data,mode='fan_out',nonlinearity='relu')
    #elif classname.find('BatchNorm') != -1:
        #m.weight.data.normal_(1.0, 0.02)
        #nn.init.xavier_normal(m.weight.data)
        #m.weight.data.fill_(1)
        #m.bias.data.zero_()
    
    
    
def main():
    '''
    # Training settings
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    # normalize X
    X_train , X_val, X_test = np.array(X_train) / 255, np.array(X_val) / 255, np.array(X_test) / 255
    # 把 y 的 string 做成 one hot encoding 形式
    label_encoder = LabelBinarizer()
    y_train_onehot = label_encoder.fit_transform(y_train)
    y_train_onehot = np.array([np.where(r==1)[0][0] for r in y_train_onehot])
    y_val_onehot = label_encoder.transform(y_val)
    y_val_onehot = np.array([np.where(r==1)[0][0] for r in  y_val_onehot])
    y_test_onehot = label_encoder.transform(y_test)
    y_test_onehot = np.array([np.where(r==1)[0][0] for r in  y_test_onehot])
    # 印一些有的沒的
    print('X_train size:', len(X_train))
    max_x = 0
    for x in X_train:
        if max_x < len(x):
            max_x = len(x)
    print('max length:',max_x)
    X_train, X_val,X_test = np.expand_dims(X_train, 1), np.expand_dims(X_val, 1),np.expand_dims(X_test, 1)
    pk.dump(X_train,open('X_train.pickle','wb'))
    pk.dump(X_val,open('X_val.pickle','wb'))
    pk.dump(y_train_onehot,open('y_train.pickle','wb'))
    pk.dump(y_val_onehot,open('y_val.pickle','wb'))
    '''
    X_train = load('training_data/X_train_ff.pickle')
    X_val = load('training_data/X_val_ff.pickle')
    y_train_onehot = load('training_data/y_train_ff.pickle')
    y_val_onehot = load('training_data/y_val_ff.pickle')
    
    X_train2 = load('training_data/X_train_ff.pickle')
    
    y_train_onehot2 = load('training_data/y_train_ff.pickle')
    
    for i in range(len(X_train2)):
        X_train2[i] = np.fliplr(X_train2[i])
    X_train = np.append(X_train,X_train2,axis = 0)
    y_train_onehot = np.append(y_train_onehot , y_train_onehot2,axis = 0)
    X_train2 = load('training_data/X_train_ff.pickle')
    for i in range(len(X_train2)):
        X_train2[i] = np.flipud(np.fliplr(X_train2[i]))
    X_train = np.append(X_train,X_train2,axis = 0)
    y_train_onehot = np.append(y_train_onehot , y_train_onehot2,axis = 0)
    
    ###vpn###
    #for i in range(len(y_train_onehot)):
    #    if(y_train_onehot[i] <= 4 or y_train_onehot[i] == 11):
    #        y_train_onehot[i] = 0
    #    else:
    #        y_train_onehot[i] = 1
    #for i in range(len(y_val_onehot)):
    #    if(y_val_onehot[i] <= 4 or y_val_onehot[i] == 11):
    #        y_val_onehot[i] = 0
    #    else:
    #        y_val_onehot[i] = 1
    traindata = Data.TensorDataset(torch.from_numpy(X_train).float(),  torch.from_numpy(y_train_onehot.astype(np.int64)))
    testdata = Data.TensorDataset(torch.from_numpy(X_val).float(),  torch.from_numpy(y_val_onehot.astype(np.int64)))
    
    use_cuda = True

    torch.manual_seed(1)

    device = torch.device("cuda")

    kwargs = {'num_workers': 1, 'pin_memory': False} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        traindata,
        batch_size=20, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        testdata,
        batch_size=30, shuffle=True, **kwargs)


    #model = Net().to(device)
    model = ResNet50().to(device)
    model.apply(weights_init)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #optimizer = optim.Adam(model.parameters())
    max_acc = -100
    min_loss = 65535
    train_rekishi = []
    test_rekishi = []
    for epoch in range(1, 24 + 1):
        
        if(epoch <=6):
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        elif(epoch >6 and epoch <= 12):
            optimizer = optim.SGD(model.parameters(), lr=0.00707, momentum=0.9)
        elif(epoch >12 and epoch <= 18):
            optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
        elif(epoch >18 and epoch <= 24):
            optimizer = optim.SGD(model.parameters(), lr=0.00354, momentum=0.9)
        else:
            optimizer = optim.SGD(model.parameters(), lr=0.0025, momentum=0.9)
        
        train( model, device, train_loader, optimizer, epoch)
        trainloss , trainacc = test(model, device, train_loader)
        loss , acc = test(model, device, test_loader)
        train_rekishi.append(trainacc)
        test_rekishi.append(acc)
        print('train_acc:',trainacc)
        
        if(loss < min_loss):
            min_loss = loss
            print('best : ', min_loss)
            pk.dump(model,open('models/res_50_aaa.pickle','wb'))
            
    #pk.dump(train_rekishi,open('result/train_his_res_101_arg.pickle','wb'))
    #pk.dump(test_rekishi,open('result/test_his_res_101_arg.pickle','wb'))
main()
