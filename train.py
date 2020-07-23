#!/usr/local/python3/bin/python3
# -*- coding: utf-8 -*-
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
# from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

import pdb
import cv2
from PIL import Image

class NeuralNet (nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        # 32x32size & 3チャネル, 隠れ層のunit数は、600
        self.conv1 = nn.Conv2d(3,  32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2,2) 
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 == nn.Linear(128*10*10, 48*48*2)
        self.pool_size = 2
        self.fc2 = nn.Linear(2304,2304)
            
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1,1228*10*10)
        x = self.fc(x)
        assert x.shape[-1] % self._pool_size == 0, \
            'Wrong input last dim size ({}) for Maxout({})'.format(x.shape[-1], self._pool_size)
        m, i = x.view(*x.shape[:-1], x.shape[-1] // self._pool_size, self._pool_size).max(-1)
        
        x = self.fc2(x)
        return x

def train(train_loader, test_loader, net, criterion, optimizer):
    #最後にlossとaccuracyのグラフを出力する為
    train_loss_list = []
    train_acc_list =  []
    val_loss_list =   []
    val_acc_list =    []

    for epoch in range(TRAIN_NUM_EPOCHS):
        #エポックごとに初期化
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0
    
        net.train()  #訓練モードへ切り替え


        
        for i, (images, labels) in enumerate(train_loader):  #ミニバッチで分割し読込み
            #viewで縦横32x32 & 3channelのimgを1次元に変換し、toでDEVICEに転送
            """
            images, labels = images.view(-1, IMG_SIZE_AND_CHANNEL).to(DEVICE), labels.to(DEVICE)
            """
            width = 96
            height = 96
            dim = (width, height)
 
            # resize image
            resize_images = np.zeros((0,3,96,96))
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            for j in range(len(images)):
                cv_image = images[j].numpy()
                cv_image = np.trancepose(cv_image,(1,2,0))
                output = cv2.resize(cv_image, dim)
                output = np.transpose(output,(2,0,1))
                output = np.reshape(output, (1,3,96,96))
                resize_images = np.append(resize_images, output, axis=0)
            images = torch.tensor(resize_images).float()
            images ,labels = images.to(DEVICE), labels.to(DEVICE)
          
            import pdb;pdb.set_trace()

            images = np.asarray(images)
            print(images.shape)
            images = cv2.resize(images, dim, interpolation = cv2.INTER_AREA)

            optimizer.zero_grad()               # 勾配をリセット
            outputs = net(images)               # 順伝播の計算
            loss = criterion(outputs, labels)   #lossの計算
            train_loss += loss.item()           #lossのミニバッチ分を溜め込む
            #accuracyをミニバッチ分を溜め込む
            #正解ラベル（labels）と予測値のtop1（outputs.max(1)）が合っている場合に1が返る
            train_acc += (outputs.max(1)[1] == labels).sum().item()
            #逆伝播の計算
            loss.backward()
            #重みの更新
            optimizer.step()
            #平均lossと平均accuracyを計算
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_acc / len(train_loader.dataset)
    

        net.eval()   #評価モードへ切り替え
        
        #評価するときに必要のない計算が走らないようtorch.no_gradを使用
        with torch.no_grad():
            for images, labels in test_loader:        
                images, labels = images.view(-1,IMG_SIZE_AND_CHANNEL).to(DEVICE),labels.to(DEVICE)
                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.max(1)[1] == labels).sum().item()
        avg_val_loss = val_loss / len(test_loader.dataset)
        avg_val_acc = val_acc / len(test_loader.dataset)
    

        # 訓練データのlossと検証データのlossとaccuracyをログ出力.
        print ("Epoch [{}/{}], Loss: {loss:.4f},val_loss: {val_loss:.4f},val_acc: {val_acc:.4f}"
               .format(epoch+1,
                       TRAIN_NUM_EPOCHS,
                       i+1,
                       loss=avg_train_loss,
                       val_loss=avg_val_loss, val_acc=avg_val_acc))

        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)

    return train_loss_list,train_acc_list,val_loss_list,val_acc_list
