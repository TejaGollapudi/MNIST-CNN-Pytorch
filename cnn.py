import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd 
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt
import torchvision.datasets as dset
import torchvision.transforms as transforms

################# DEFINING THE NEURAL NETWORKS MODEL ##############################
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        #Conv2d parameters (n_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(1, 32, 3,padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3,padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3=nn.Conv2d(64,32,5,padding=2)
        self.conv3_bn=nn.BatchNorm2d(32)
       # self.conv4=nn.Conv2d(32,28,5,padding=2)
        #self.conv4_bn=nn.BatchNorm2d(28)
        #self.conv3=nn.Conv2d(16,24,3,2)
        self.fc1 = nn.Linear(32 * 3 * 3, 140)
        self.fc1_bn = nn.BatchNorm1d(140)
        self.fc2=nn.Linear(140,50)
        self.fc2_bn=nn.BatchNorm1d(50)
        self.fc3 = nn.Linear(50, 10)

    #defining the forward pass of the network 
    def forward(self, x):
        # Max pooling over a (2, 2) window
        # leaky_relu(input, negative_slope=0.01, inplace=False):
        x = F.max_pool2d(F.relu(self.conv1_bn(self.conv1(x))), (2, 2),2)
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2_bn(self.conv2(x))), 2,2)
        x = F.max_pool2d(F.relu(self.conv3_bn(self.conv3(x))), 2,2)
        #print('maxpool3 output',x.size())
        x = x.view(-1, self.num_flat_features(x))
        #print('flatten output',x.size())
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        x=F.softmax(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

###################################LOAD DATASET##################################################

#transform is used for data auggmentation ,normalization and tensor conversion
trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (1.0,))])

train_set = dset.MNIST(root='./data', train=True,transform=trans,  download=True)
test_set= dset.MNIST(root='./data', train=False,transform=trans,  download=True)

batch_size = 200

#Dataloader is used to split data into mini batches

train_loader = torch.utils.data.DataLoader( dataset=train_set, batch_size=batch_size,shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True )

print('.........Dataset Loaded ......')

##################################################################################################

ploss=[] #stores loss info
piter=[]
net = Net() 
#prints the basic convnet structure
print('model summary',net)
#print details of learnable parameters
params = list(net.parameters())
print(len(params))
print(params[0].size())  

######################define optimizer###################################
#optimizer = optim.Adadelta(net.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0) #adadelta
optimizer=optim.Adam(net.parameters())


######################checking model with a randomly intialized tensor###########################
#this is done to check if model is functioning or not

input = Variable(torch.randn(1, 1, 28, 28))
target = Variable(torch.arange(1, 11)) 
 # a dummy target, for example
print(target.size())
target = target.view(1, -1) 
 # make it the same shape as output
print(target.size())

#############define loss function#################
criterion = nn.CrossEntropyLoss()



###########train function###################
def train():
    k=0

    for i in range(2):
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()   # clear the gradient buffers
            x, target = Variable(x), Variable(target)
            #target = target.view(1, -1)
            #target=target.float()
            output = net(x)
            #print('output',output)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step() 
            print('iteration',i,'batch_idx',batch_idx,'loss',loss,'loss data',loss.data[0])
            k=k+1
            ploss.append(loss.data[0])
            piter.append(k)
            print(k)

###########train called###################
train()



###############Plot##############
plt.plot(piter,ploss,'o-',color='g',label='L C')
plt.title('epoch vs Loss')
plt.ylabel('loss')
plt.xlabel('epoch ')
#plt.legend(loc='best')
plt.show()
#############test function##############
def test():
    net.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

################call test function ######################
print('...............test function called.............................')
test()

