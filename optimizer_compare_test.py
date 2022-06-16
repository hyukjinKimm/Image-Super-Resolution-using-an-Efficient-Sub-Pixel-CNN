import os, cv2
import pickle
import numpy as np
from collections import OrderedDict
from common.function import *
from common.layer import *
from common.preprocess import *
from common.optimizer import *
import sr_net

test_model ={}
train_loss = {}
# 가중치 값을 가져옵니다.
for key in ['SGD', 'Adam', 'AdaGrad', 'Momentum']:
    with open("./weight/optimizer_compare/" + key +"/W1.pickle","rb") as fr:
        W1 = pickle.load(fr)
    with open("./weight/optimizer_compare/" + key +"/W2.pickle","rb") as fr:
        W2 = pickle.load(fr)
    with open("./weight/optimizer_compare/" + key +"/W3.pickle","rb") as fr:
        W3 = pickle.load(fr)
    with open("./weight/optimizer_compare/" + key +"/W4.pickle","rb") as fr:
        W4 = pickle.load(fr)
    with open("./weight/optimizer_compare/" + key +"/train_loss.pickle","rb") as fr:
        loss = pickle.load(fr)
    test_model[key] = sr_net.SR_Net(W1, W2, W3, W4)
    train_loss[key] = loss

base_path = './data/'
img_path = os.path.join(base_path, 'img_align_celeba')
target_img_path = os.path.join(base_path, 'processed')
eval_list = np.loadtxt(os.path.join(base_path, 'list_eval_partition.csv'), dtype=str, delimiter=',', skiprows=1)

train_size = 162770
validation_size = 182637
test_size = 202599 

size = 10
batch_mask = np.random.randint(validation_size, test_size, size = size)
batch_list = eval_list[batch_mask] 
img_files = [os.path.join(img_path, batch_list[idx][0]) for idx in range(size)]
images = np.array([cv2.imread(img_files[idx]) for idx in range(size)])
# images = (size, 216, 278, 3)

HR_images = {}  

for key in ['SGD', 'Adam', 'AdaGrad', 'Momentum']:
    LR_images, HR_images[key], answer_images = test_model[key].test(images, size)

for i in range(size):
    img = cv2.cvtColor(LR_images[i], cv2.COLOR_BGR2RGB)
    plt.imshow(img) 
    plt.savefig('./images/optimizer_compare_images/LR/LR_' + str(i)+ '.png')

    img = cv2.cvtColor(answer_images[i], cv2.COLOR_BGR2RGB)
    plt.imshow(img) 
    plt.savefig('./images/optimizer_compare_images/ANSWER/ANSWER_' + str(i)+ '.png')

for key in ['SGD', 'Adam', 'AdaGrad', 'Momentum']:
    for i in range(size):
        img = cv2.cvtColor(HR_images[key][i], cv2.COLOR_BGR2RGB)
        plt.imshow(img) 
        plt.savefig('./images/optimizer_compare_images/' + key +'/HR_' + str(i)+ '.png')
        
