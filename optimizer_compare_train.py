import os, cv2
import pickle
import numpy as np
from collections import OrderedDict
from common.function import *
from common.layer import *
from common.preprocess import *
from common.optimizer import *
import sr_net

# 1. 실험용 설정==========
optimizers = {}
optimizers['SGD'] = SGD()
optimizers['Momentum'] = Momentum()
optimizers['AdaGrad'] = AdaGrad()
optimizers['Adam'] = Adam()

networks = {}
train_loss = {}
channel = []
channel = ['B', 'G', 'R']

#가중치 값을 가져와 이어서 학습합니다
try:
    with open('./weight/optimizer_compare/index.pickle', 'rb') as fr:
        idx = pickle.load(fr)
    print('학습을 재개 합니다')
    for key in optimizers.keys():
        W = {}
        for th in ['W1', 'W2', 'W3', 'W4']:
            with open("./weight/optimizer_compare/" + key + "/" + th + ".pickle","rb") as fr:
                W[th] = pickle.load(fr)
        networks[key] = sr_net.SR_Net(W['W1'], W['W2'], W['W3'], W['W4'])
        with open("./weight/optimizer_compare/" + key + "/train_loss.pickle","rb") as fr:
            train_loss[key] = pickle.load(fr)
except FileNotFoundError:
    print("새로운 네트워크를 만듭니다.")
    for key in optimizers.keys():
        idx = 0
        networks[key] = sr_net.SR_Net()
        train_loss[key] = []

base_path = './data/'
img_path = os.path.join(base_path, 'img_align_celeba')
target_img_path = os.path.join(base_path, 'processed')
eval_list = np.loadtxt(os.path.join(base_path, 'list_eval_partition.csv'), dtype=str, delimiter=',', skiprows=1)

train_size = 162770
validation_size = 182637
test_size = 202599 

iters_num =  1000
batch_size = 10# 미니배치 크기

for i in range(idx, iters_num):
    for key in optimizers.keys():
        # images = (batch_size, 218, 178, 3)
        batch_mask = np.random.choice(train_size, batch_size)
        batch_list = eval_list[batch_mask] 
        img_files = [os.path.join(img_path, batch_list[idx][0]) for idx in range(batch_size)]
        images = np.array([cv2.imread(img_files[idx]) for idx in range(batch_size)])
        (x_batch, t_batch) = preprocess(images)
        print("optimizer : " + key)
        for ch in range(3):
            t_batch_tmp = t_batch[:, :, :,ch].reshape((batch_size, 176, 176, 1)).transpose(0, 3, 1, 2).astype(np.float64) 
            x_batch_tmp = x_batch[:, :, :,ch].reshape((batch_size, 44, 44, 1)).transpose(0, 3, 1, 2).astype(np.float64) / 255
            grads = networks[key].gradient(x_batch_tmp, t_batch_tmp)
            optimizers[key].update(networks[key].params, grads)
            loss = networks[key].loss(x_batch_tmp, t_batch_tmp)
            print(str(i+1) + "번째 배치의 " + channel[ch] + "채널 학습 loss 값 " +  str(loss)) 
            train_loss[key].append(loss)

    if i % 3 == 0:
        with open('./weight/optimizer_compare/index.pickle', 'wb') as f:
            pickle.dump(i, f, pickle.HIGHEST_PROTOCOL)
        for key in optimizers.keys():
            with open('./weight/optimizer_compare/' + key + '/train_loss.pickle', 'wb') as f:
                pickle.dump(train_loss[key], f, pickle.HIGHEST_PROTOCOL)
            for th in ['W1', 'W2', 'W3', 'W4']:
                with open('./weight/optimizer_compare/' + key + '/' + th + '.pickle', 'wb') as f:
                    pickle.dump(networks[key].params[th], f, pickle.HIGHEST_PROTOCOL)
    print(str(i) + '번째 학습 완료')
    




