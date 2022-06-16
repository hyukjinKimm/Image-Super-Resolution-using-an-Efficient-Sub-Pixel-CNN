import os, cv2
import pickle
import numpy as np
from collections import OrderedDict
from common.function import *
from common.layer import *
from common.preprocess import *
from common.optimizer import *
import sr_net

# 가중치 값을 가져와 이어서 학습합니다.
with open("./weight/train/index.pickle","rb") as fr:
    index = pickle.load(fr)
with open("./weight/train/W1.pickle","rb") as fr:
    W1 = pickle.load(fr)
with open("./weight/train/W2.pickle","rb") as fr:
    W2 = pickle.load(fr)
with open("./weight/train/W3.pickle","rb") as fr:
    W3 = pickle.load(fr)
with open("./weight/train/W4.pickle","rb") as fr:
    W4 = pickle.load(fr)
with open("./weight/train/train_loss.pickle","rb") as fr:
        train_loss = pickle.load(fr)
        
#옵티마이저 로 아담을 사용합니다.
optimizers = {}
optimizers['Adam'] = Adam()
train_net = sr_net.SR_Net(W1, W2, W3, W4)

channel = ['B', 'G', 'R']



base_path = './data/'
img_path = os.path.join(base_path, 'img_align_celeba')
target_img_path = os.path.join(base_path, 'processed')
eval_list = np.loadtxt(os.path.join(base_path, 'list_eval_partition.csv'), dtype=str, delimiter=',', skiprows=1)
img_sample = cv2.imread(os.path.join(img_path, eval_list[0][0]))

train_size = 162770
validation_size = 182637
test_size = 202599 

iters_num =  50000
batch_size = 10# 미니배치 크기

for i in range(index, iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    batch_list = eval_list[batch_mask] 

    img_files = [os.path.join(img_path, batch_list[idx][0]) for idx in range(batch_size)]
    images = np.array([cv2.imread(img_files[idx]) for idx in range(batch_size)])
    # images = (batch_size, 218, 178, 3)

    (x_batch, t_batch) = preprocess(images)
    for ch in range(3):
        t_batch_tmp = t_batch[:, :, :,ch].reshape((batch_size, 176, 176, 1)).transpose(0, 3, 1, 2).astype(np.float64) 
        x_batch_tmp = x_batch[:, :, :,ch].reshape((batch_size, 44, 44, 1)).transpose(0, 3, 1, 2).astype(np.float64) / 255

        grad = train_net.gradient(x_batch_tmp, t_batch_tmp)
        optimizers['Adam'].update(train_net.params, grad)
        loss = train_net.loss(x_batch_tmp, t_batch_tmp)
        print(str(i + 1) + "번째 배치의 " + channel[ch] + "채널 학습 loss 값 " +  str(loss)) 
        train_loss.append(loss)
        

    with open('./weight/train/index.pickle', 'wb') as f:
        pickle.dump(i, f, pickle.HIGHEST_PROTOCOL)
    with open('./weight/train/train_loss.pickle', 'wb') as f:
        pickle.dump(train_loss, f, pickle.HIGHEST_PROTOCOL)    
    for th in ['W1', 'W2', 'W3', 'W4']:
        with open("./weight/train/" + th + '.pickle', 'wb') as f:
            pickle.dump(train_net.params[th], f, pickle.HIGHEST_PROTOCOL)