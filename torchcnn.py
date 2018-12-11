import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torchvision import datasets, transforms


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
    directory = 'data'
    todo_list = gen_todo_list(directory, check = check)
    ### ver 1 ###
    train_rate = 0.64
    val_rate = 0.16
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
        tmpX= processX(tmpX)
        
        #random.shuffle(tmpX)
        
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


def processX(X):
    if True:
        
        X = np.array(X)
        lens = [len(x) for x in X] 
        maxlen = 1500
        tmpX = np.zeros((len(X), maxlen))
        mask = np.arange(maxlen) < np.array(lens)[:,None]
        tmpX[mask] = np.concatenate(X)
        return tmpX
        
        '''cnt =0
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
        return tmpX , tmpY'''
    else:
        for i, x in enumerate(X):
            tmp_x = np.zeros((1500,))
            tmp_x[:len(x)] = x
            X[i] = tmp_x
        return X

    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        
        #self.conv1_drop = nn.Dropout2d(p=0.05)
        
        self.conv11 = nn.Conv1d(1, 200, kernel_size=5, padding = 1)
        self.conv11_bn = nn.BatchNorm1d(200)        
        self.conv12 = nn.Conv1d(200, 100, kernel_size=4, padding = 1)
        self.conv12_bn = nn.BatchNorm1d(100)        
        
        
        self.fc0 = nn.Linear(74800, 600)
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
        
        x =   self.conv11(x) 
        x = F.relu(self.conv11_bn(x)  )
        x = F.dropout(x, training=self.training,p=0.4)
        x = self.conv12(x)
        x = F.relu( self.conv12_bn( x ) )
        x = F.max_pool1d(  x,2)
        x = F.dropout(x, training=self.training,p=0.4)
          
        
        x = x.view( x.size(0),-1)
       
        x = F.relu(self.fc0_bn( self.fc0(x) ))
        
        
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.dropout(x, training=self.training,p=0.4)
        
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.dropout(x, training=self.training,p=0.4)
        
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = F.dropout(x, training=self.training,p=0.4)
        
        x = F.relu(self.fc4_bn( self.fc4(x) ))
        x = F.dropout(x, training=self.training,p=0.4)
        
        x = self.fc5(x)
        x = F.relu( self.fc5_bn( x) )
        x = F.dropout(x, training=self.training,p=0.4)
        x = self.fc6(x)
        x = F.relu( self.fc6_bn( x) )
        x = F.dropout(x, training=self.training,p=0.4)
        
        
        x = self.fc7( x)
        return F.log_softmax(x, dim=1)
    
    
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    #weights = np.array([1,2,1,1,5,5,5,2,2,5,1,0.7])
    #weights= torch.from_numpy(weights).float().cuda()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target )
        loss.backward()
        optimizer.step()
        if batch_idx % 300 == 0:
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
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    
    
    
def main():
    
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
    
    traindata = Data.TensorDataset(torch.from_numpy(X_train).float(),  torch.from_numpy(y_train_onehot.astype(np.int64)))
    testdata = Data.TensorDataset(torch.from_numpy(X_val).float(),  torch.from_numpy(y_val_onehot.astype(np.int64)))
    
    use_cuda = True

    torch.manual_seed(1)

    device = torch.device("cuda")

    kwargs = {'num_workers': 1, 'pin_memory': False} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        traindata,
        batch_size=100, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        testdata,
        batch_size=150, shuffle=True, **kwargs)


    model = Net().to(device)
    model.apply(weights_init)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #optimizer = optim.Adam(model.parameters())
    max_acc = -100
    min_loss = 1000000
    train_rekishi = []
    test_rekishi = []
    for epoch in range(1, 45 + 1):
        
        if(epoch <=15):
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        elif(epoch >15 and epoch <= 30):
            optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
        else:
            optimizer = optim.SGD(model.parameters(), lr=0.0025, momentum=0.9)
        
        train( model, device, train_loader, optimizer, epoch)
        trainloss,trainacc = test(model, device, train_loader)
        loss,acc = test(model, device, test_loader)
        train_rekishi.append(trainacc)
        test_rekishi.append(acc)
        print('train_acc:',trainacc)
        if(loss<min_loss):
            min_loss = loss 
            print('best : ', min_loss)
            pk.dump(model,open('models/perpacket.pickle','wb'))
    pk.dump(train_rekishi,open('result/train_his_perpacket.pickle','wb'))
    pk.dump(test_rekishi,open('result/test_his_perpacket.pickle','wb'))
main()
