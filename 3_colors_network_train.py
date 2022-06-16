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

try:
    with open('./weight/color_weight/index.pickle', 'rb') as fr:
        idx = pickle.load(fr)
    print('학습을 재개 합니다')
    for c in ['B', 'G', 'R']:
        networks[c] = {} 
        train_loss[c] = {}
        W = {}
        for th in ['W1', 'W2', 'W3', 'W4']:
            with open("./weight/color_weight/"  + c + '/' + th + ".pickle","rb") as fr:
                W[th] = pickle.load(fr)
        networks[c]  = sr_net.SR_Net(W['W1'], W['W2'], W['W3'], W['W4'])
        with open("./weight/color_weight/" + c + "/train_loss.pickle","rb") as fr:
            train_loss[c] = pickle.load(fr)

except FileNotFoundError:
    for c in ['B', 'G', 'R']:
        idx = 0
        networks[c] = sr_net.SR_Net()
        train_loss[c] = []


base_path = './data/'
img_path = os.path.join(base_path, 'img_align_celeba')
target_img_path = os.path.join(base_path, 'processed')
eval_list = np.loadtxt(os.path.join(base_path, 'list_eval_partition.csv'), dtype=str, delimiter=',', skiprows=1)
img_sample = cv2.imread(os.path.join(img_path, eval_list[0][0]))


train_size = 162770
validation_size = 182637
test_size = 202599 

iters_num =  25000
batch_size = 10# 미니배치 크기

for i in range(idx, iters_num):


    # images = (batch_size, 218, 178, 3)
    batch_mask = np.random.choice(train_size, batch_size)
    batch_list = eval_list[batch_mask] 
    img_files = [os.path.join(img_path, batch_list[idx][0]) for idx in range(batch_size)]
    images = np.array([cv2.imread(img_files[idx]) for idx in range(batch_size)])

    (x_batch, t_batch) = preprocess(images)

    for k, c in enumerate(['B', 'G', 'R']):
        t_batch_tmp = t_batch[:, :, :,k].reshape((batch_size, 176, 176, 1)).transpose(0, 3, 1, 2).astype(np.float64) 
        x_batch_tmp = x_batch[:, :, :,k].reshape((batch_size, 44, 44, 1)).transpose(0, 3, 1, 2).astype(np.float64) / 255
        grads = networks[c].gradient(x_batch_tmp, t_batch_tmp)
        optimizers['Adam'].update(networks[c].params, grads)
        loss = networks[c].loss(x_batch_tmp, t_batch_tmp)
        print( c + "  채널의 로스값 " + str(loss))
        train_loss[c].append(loss)
    
    if i % 3 == 0:
        for key in optimizers.keys():
            with open('./weight/color_weight/index.pickle', 'wb') as f:
                pickle.dump(i, f, pickle.HIGHEST_PROTOCOL)
            for c in ['B', 'G', 'R']:
                with open('./weight/color_weight/' + c  +'/train_loss.pickle', 'wb') as f:
                    pickle.dump(train_loss[c], f, pickle.HIGHEST_PROTOCOL)
                for th in ['W1', 'W2', 'W3', 'W4']:
                    with open('./weight/color_weight/'  + c + '/' + th + '.pickle', 'wb') as f:
                        pickle.dump(networks[c].params[th], f, pickle.HIGHEST_PROTOCOL)
    print(str(i) + '번째 학습 완료')
    




