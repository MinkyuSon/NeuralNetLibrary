# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 08:44:36 2020

@author: minky
"""

#arr.reshape(-1,1) shapes array to column vector
import numpy as np
import struct as st
from datetime import datetime

class NeuralNetLayer():
    def __init__():
        pass
    
    def forProp():
        pass
    
    def backProp():
        pass
    
class LinearLayer:
    def __init__(self, PrevLayerSize, LayerSize):
        self.size = (PrevLayerSize, LayerSize)
        self.Weight = np.random.randn(self.size[1],self.size[0])
        self.Bias = np.random.randn(self.size[1]).reshape(-1,1)
        
        self.WeightGradients = np.zeros(self.Weight.shape)
        self.WeightGradientsCount = 0
        self.BiasGradients = np.zeros(self.Bias.shape)
        self.BiasGradientsCount = 0
        
    def forwardProp(self, a):
        self.a = np.array(a).reshape(-1,1)
        self.z = (np.matmul(self.Weight, self.a)+self.Bias).reshape(-1,1)
        return self.z
    
    def backProp(self):
        return self.Weight
    
    def WeightGradient(self, dJdz): #eta is the learning rate
        self.dzdw = np.zeros((self.size[1], self.size[0]*self.size[1]))
        for k in range(self.size[1]):
            self.dzdw[k][k*self.size[0]:(k+1)*self.size[0]] = self.a.transpose()
        self.dJdw = np.matmul(dJdz.T,self.dzdw).reshape(self.size[1],self.size[0])
        self.dJdb = dJdz #turns out to be the same
        self.WeightGradients += self.dJdw
        self.BiasGradients += self.dJdb
        self.WeightGradientsCount += 1
        self.BiasGradientsCount += 1
        

    def updateWeight(self, eta):
        self.Weight -= (eta/self.WeightGradientsCount)*self.WeightGradients
        self.Bias -= (eta/self.BiasGradientsCount)*self.BiasGradients
        
        self.WeightGradients = np.zeros(self.Weight.shape)
        self.WeightGradientsCount = 0
        self.BiasGradients = np.zeros(self.Bias.shape)
        self.BiasGradientsCount = 0
        
        return (self.Weight, self.Bias)

                
class SoftMaxLayer:
    def __init__(self, *LayerSize):
        self.size = LayerSize[0]
    
    def forwardProp(self, z):
        self.z = np.array(z).reshape(-1,1)
        SumExp = np.nan_to_num(sum(np.exp(z))) #in case we get np.inf
        self.a = (np.nan_to_num(np.exp(z))/SumExp).reshape(-1,1)
        return self.a
    
    def backProp(self):
        self.dadz = (np.diag(self.a.T[0])-self.a@self.a.T)
        return self.dadz

class SoftPlusLayer:
    def __init__(self, *LayerSize):
        self.size = LayerSize[0]
        
    def forwardProp(self, z):
        self.z = np.array(z).reshape(-1,1)
        self.a = np.log(1+np.nan_to_num(np.exp(self.z))).reshape(-1,1)
        return self.a
    
    def backProp(self):
        self.dadz = np.diag(1/(1+np.nan_to_num(np.exp(-self.z))).T[0])
        return self.dadz

class CostLayer:
    def __init__(self, *LayerSize):
        self.eps = np.exp(-100)
        self.size = LayerSize[0]
    
    def forwardProp(self, a, y):
        self.a = np.array(a).reshape(-1,1)
        self.y = np.array(y).reshape(-1,1)
        return -(np.log(a+self.eps).T@y + np.log(1-a-self.eps)@(1-y))
    
    def backProp(self):
        self.dJda = -1*np.where(self.y==1, 1/(self.a+self.eps), 1/(self.a+self.eps-1)).reshape(-1,1)
        return self.dJda.reshape(-1,1)
    
class LayerSequence:
    LayerDict = {'R': SoftPlusLayer, 'S': SoftMaxLayer, 'C': CostLayer, 'L': LinearLayer}

    def __init__(self, LayerList, LayerSize): #LayerList is the string representing the order of label. LayerSize is a list of pair for each layers
        self.LayerList = LayerList
        assert LayerList[-1] == 'C', 'last layer must be costLayer'
        self.Network = []
        self.NetworkLength = len(LayerList)
        for k in range(self.NetworkLength):
            self.Network.append(self.LayerDict[LayerList[k]](*LayerSize[k]))

    def forwardProp(self, a, y):
        for k in range(self.NetworkLength-1): #not including the costLayer
            a = self.Network[k].forwardProp(a)
        self.cost = self.Network[-1].forwardProp(a, y)
        return a, self.cost

    def backProp(self):
        self.CountWeight = self.LayerList.count('L')
        self.backPropWeights = [0 for k in range(self.CountWeight)]
        weightLayerIndex = 0
        self.backPropChain = self.Network[-1].backProp() #CostLayer
        
        for k in range(-2, -self.NetworkLength-1, -1): #start at the final layer and backpropogate
            if self.LayerList[k] == 'L':
                self.backPropWeights[weightLayerIndex] = (k, np.copy(self.backPropChain))
                weightLayerIndex += 1
            self.backPropChain = (self.backPropChain.T@self.Network[k].backProp()).T
            
        for k in self.backPropWeights:
            self.Network[k[0]].WeightGradient(k[1])

    def updateWeight(self, eta): #eta is the learning rate
        self.WeightsList = {}
        self.BiasesList = {}
        for k in self.backPropWeights: #store updated weights for saving
            self.WeightsList[str(k[0])], self.BiasesList[str(k[0])] = self.Network[k[0]].updateWeight(eta)

    def SaveWeight(self, FileName):
        np.savez(FileName+'_Weight.npz', **self.WeightsList)
        np.savez(FileName+'_Bias.npz', **self.BiasesList)
    
    def LoadWeight(self, FileName):
        with np.load(FileName+'_Weight.npz') as WeightsFile:
            for k in WeightsFile.keys():
                index = int(k)
                assert self.Network[index].Weight.shape == WeightsFile[k].shape, 'the shape of loaded matrix should match'
                self.Network[index].Weight = WeightsFile[k]
        with np.load(FileName+'_Bias.npz') as BiasesFile:
            for k in BiasesFile.keys():
                index = int(k)
                assert self.Network[index].Bias.shape == BiasesFile[k].shape, 'the shape of loaded matrix should match'
                self.Network[index].Bias = BiasesFile[k]


            
##################################################################################
            

filename = {'train_images' : 'train-images.idx3-ubyte' ,'train_labels' : 'train-labels.idx1-ubyte', 'test_images' : 't10k-images.idx3-ubyte', 'test_labels' : 't10k-labels.idx1-ubyte'}

with open(filename['train_images'],'rb') as train_imagesfile:
    train_imagesfile.seek(0)
    magic = st.unpack('>4B',train_imagesfile.read(4))
    nImg = st.unpack('>I',train_imagesfile.read(4))[0] #num of images
    nR = st.unpack('>I',train_imagesfile.read(4))[0] #num of rows
    nC = st.unpack('>I',train_imagesfile.read(4))[0] #num of column
    nBytesTotal = nImg*nR*nC*1 #since each pixel data is 1 byte
    train_images_array = np.zeros((nImg,nR,nC))
    train_images_array = 255 - np.asarray(st.unpack('>'+'B'*nBytesTotal,train_imagesfile.read(nBytesTotal))).reshape((nImg,nR,nC))
    train_imagesfile.seek(16)
    train_images_array_train = np.zeros((nImg,nR*nC))# feeding into the neuralNet
    train_images_array_train = 255 - np.asarray(st.unpack('>'+'B'*nBytesTotal,train_imagesfile.read(nBytesTotal))).reshape((nImg,nR*nC))

with open(filename['test_images'],'rb') as test_imagesfile:
    test_imagesfile.seek(0)
    magic = st.unpack('>4B',test_imagesfile.read(4))
    nImg = st.unpack('>I',test_imagesfile.read(4))[0] #num of images
    nR = st.unpack('>I',test_imagesfile.read(4))[0] #num of rows
    nC = st.unpack('>I',test_imagesfile.read(4))[0] #num of column
    nBytesTotal = nImg*nR*nC*1 #since each pixel data is 1 byte
    test_images_array = np.zeros((nImg,nR,nC))
    test_images_array = 255 - np.asarray(st.unpack('>'+'B'*nBytesTotal,test_imagesfile.read(nBytesTotal))).reshape((nImg,nR,nC))
    test_imagesfile.seek(16)
    test_images_array_train = np.zeros((nImg,nR*nC))#feeding into the neuralNet
    test_images_array_train = 255 - np.asarray(st.unpack('>'+'B'*nBytesTotal,test_imagesfile.read(nBytesTotal))).reshape((nImg,nR*nC))

with open(filename['train_labels'], 'rb') as train_labelsfile:
    train_labelsfile.seek(0)
    magic = st.unpack('>4B',train_labelsfile.read(4))
    nLabel = st.unpack('>I',train_labelsfile.read(4))[0] #num of labels
    train_labels_array = np.zeros((nLabel))
    nBytesTotal = nLabel*1 #since each label data is 1 byte
    train_labels_array = np.asarray(st.unpack('>'+'B'*nBytesTotal,train_labelsfile.read(nBytesTotal))).reshape((nLabel))

with open(filename['test_labels'], 'rb') as test_labelsfile:
    test_labelsfile.seek(0)
    magic = st.unpack('>4B',test_labelsfile.read(4))
    nLabel = st.unpack('>I',test_labelsfile.read(4))[0] #num of labels
    test_labels_array = np.zeros((nLabel))
    nBytesTotal = nLabel*1 #since each label data is 1 byte
    test_labels_array = np.asarray(st.unpack('>'+'B'*nBytesTotal,test_labelsfile.read(nBytesTotal))).reshape((nLabel))
    
train_images_array = train_images_array/256
train_images_array_train = train_images_array_train/256
test_images_array = test_images_array/256
test_images_array_train = test_images_array_train/256

def LabelEncode(Label, length):
    ANS = np.zeros(length)
    ANS[Label] = 1
    return ANS

"""
k = 4
imgplot = plt.imshow(train_images_array[k], cmap = 'gray')
print(train_labels_array[k])
"""

print('Data Loaded')

HandWriteRecogNet = LayerSequence('LRLRLSC', ((784,16), (16,16), (16,16), (16,16), (16,10), (10,10), (10,1)))
#HandWriteRecogNet.LoadWeight('HandWriting')

a = train_images_array_train[1]
y = np.zeros(10)
y[train_labels_array[1]] = 1



A = np.random.rand(784).reshape(-1,1)
y = [1,0,0,0,0,0,0,0,0,0]
HandWriteRecogNet.forwardProp(A,y)


epoch = 1000
Break = False
for k in range(epoch):
    shuffle = np.arange(60000)
    np.random.shuffle(shuffle)
    for i in range(100):
        for j in range(600):
            a = train_images_array_train[shuffle[600*i+j]]
            y = LabelEncode(train_labels_array[shuffle[600*i+j]], 10)
            ANS = HandWriteRecogNet.forwardProp(a,y)[0]
#            print(ANS)
            HandWriteRecogNet.backProp()
            if np.isnan(ANS[-1][0]):
                Break = True
                break
        if Break:
            break
        HandWriteRecogNet.updateWeight(0.05)
#        if input('Continue?: ').lower()[0] == 'n':
#            break
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    HandWriteRecogNet.SaveWeight('HandWriting_'+str(k)+'_'+now)
    print(now, k, epoch)

