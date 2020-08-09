import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from shallow_test4 import MLPNet

 
import numpy as np
import cv2
import pdb
import glob
import matplotlib.pyplot as plt

def softmax(targets): #softmaxにより確率分布を取得
    targets_max = np.max(targets)
    expp = np.exp(targets-targets_max)
    total = np.sum(expp)
    return expp/total
TRAIN_NUM_EPOCHS = 50   #50エポック


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#images = glob.glob('./train/COCO_train2014_000000000009.jpg')
cv_images = cv2.imread('./train/COCO_train2014_000000000009.jpg')
cv_images = cv2.resize(cv_images,(96,96))
cv_images = np.transpose(cv_images,(2,0,1))
t_images = torch.Tensor(cv_images)
t_images = t_images.to(DEVICE)

#import pdb;pdb.set_trace()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
#====== モデル(ネットワーク)　設定======
net = MLPNet().to(device = device)
 
#====== ロード =======
#import pdb;pdb.set_trace()
net.load_state_dict(torch.load("./checkpoint/yanakakami2.pth"))
#import pdb;pdb.set_trace()
 
#====== 推論 ======
criterion = nn.MSELoss()
# 以下のparameterの妥当性は理解していません
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)


#最後にlossとaccuracyのグラフを出力する為
train_loss_list = []
train_acc_list =  []
val_loss_list =   []
val_acc_list =    []

files = sorted(glob.glob('./maps/train/*.png')) #hontoha"val"womoltutekitehyoukasuru
files_rgb = sorted(glob.glob('./train/*.jpg'))  #konnkaihanai


batch = 20
iteration = int(len(files)/20)

for epoch in range(TRAIN_NUM_EPOCHS):
    #エポックごとに初期化
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0
    
    net.eval()  #huyouinagyakudennpannwosinaitame


        
    # for i, (images, labels) in enumerate(train_loader):  #ミニバッチで分割し読込み
    for i in range(iteration):   
        #viewで縦横32x32 & 3channelのimgを1次元に変換し、toでDEVICEに転送
        
        width = 96
        height = 96
        dim = (width, height)
 
        # resize image
        resize_images = np.zeros((0,3,96,96))
        resize_saliency = np.zeros((0,1,48,48))

        """
        import pdb;pdb.set_trace()
        """

        for j in range(batch):
            cv_image = cv2.imread(files_rgb[batch*i + j])
            cv_image = cv2.resize(cv_image,(96,96))
            cv_image = np.transpose(cv_image,(2,0,1))
            cv_image = np.reshape(cv_image, (1,3,96,96))
            cv_saliency = cv2.imread(files[batch*i + j],cv2.IMREAD_GRAYSCALE)
            cv_saliency = cv2.resize(cv_saliency,(48,48))
            cv_saliency = cv_saliency/np.max(cv_saliency)
            #cv_saliency = np.transpose(cv_saliency,(2,0,1))
            cv_saliency = np.reshape(cv_saliency, (1,1,48,48))
           
            resize_images = np.append(resize_images, cv_image, axis=0)
            resize_saliency = np.append(resize_saliency, cv_saliency, axis=0)    
        resize_images = torch.Tensor(resize_images)
        resize_saliency = torch.Tensor(resize_saliency)

        resize_images, resize_saliency = resize_images.to(DEVICE), resize_saliency.to(DEVICE)

        
        optimizer.zero_grad()               # 勾配をリセット
        outputs = net(resize_images)
        

    
        
        loss = criterion(outputs,resize_saliency)   #lossの計算
        print(i + 1)
        print(loss.item())
        train_loss += loss.item() 
        
        inputs_ii = resize_images.cpu().detach().numpy()
        inputs_ii = inputs_ii.transpose(0,2,3,1)
        inputs_ii = inputs_ii.astype(np.uint8)####cv2のRGB画像を用意
        out_sals = outputs.cpu().detach().numpy()
        salmaps_true_np = resize_saliency.cpu().detach().numpy()

        #import pdb;pdb.set_trace()

        for sal in range(batch):
            inputs_yanaka = cv2.resize(inputs_ii[sal], (640, 480))
        
            #saliency mapの真値と予測値をsoftmaxにより確率化
            out_sal = softmax(out_sals[sal][0])#softmax
            out_sal = (out_sal-np.min(out_sal))/(np.max(out_sal)-np.min(out_sal))*255 #正規化
            #import pdb;pdb.set_trace()
            out_sal = cv2.resize(out_sal, (640, 480))
            out_sal = out_sal.astype(np.uint8)
            salmap_true_np = softmax(salmaps_true_np[sal][0])#softmax
            salmap_true_np = (salmap_true_np-np.min(salmap_true_np))/(np.max(salmap_true_np)-np.min(salmap_true_np))*255 #正規化
            
            salmap_true_np = cv2.resize(salmap_true_np, (640, 480))
            salmap_true_np = salmap_true_np.astype(np.uint8)
            
            #import pdb;pdb.set_trace()
            jet_map1 = cv2.applyColorMap(out_sal, cv2.COLORMAP_JET)
            jet_map1 = cv2.addWeighted(inputs_yanaka, 0.5, jet_map1, 0.5, 0)
            jet_map2 = cv2.applyColorMap(salmap_true_np, cv2.COLORMAP_JET)
            jet_map2 = cv2.addWeighted(inputs_yanaka, 0.5, jet_map2, 0.5, 0)
            jet_concat = np.concatenate([jet_map2, jet_map1], axis=0)
            #cv2.imwrite(‘./outputs_val/out_sal_val’ + str(i*args.batch+sal) +‘.png’, jet_map1)
            cv2.imwrite('./outputs_val/out_sal_cat' + str(i*batch+sal) +'.png', jet_concat)
        #import pdb;pdb.set_trace()         #lossのミニバッチ分を溜め込む
    #accuracyをミニバッチ分を溜め込む
    #逆伝播の計算
    #loss.backward()
    avg_train_loss = train_loss / iteration
