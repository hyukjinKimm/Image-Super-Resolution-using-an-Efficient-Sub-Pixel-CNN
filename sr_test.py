import os, cv2
import pickle
import numpy as np
from collections import OrderedDict
from common.function import *
from common.layer import *
from common.preprocess import *
from common.optimizer import *
import sr_net


with open("./weight/train/W1.pickle","rb") as fr:
    W1 = pickle.load(fr)
with open("./weight/train/W2.pickle","rb") as fr:
    W2 = pickle.load(fr)
with open("./weight/train/W3.pickle","rb") as fr:
    W3 = pickle.load(fr)
with open("./weight/train/W4.pickle","rb") as fr:
    W4 = pickle.load(fr)

test_model = sr_net.SR_Net(W1, W2, W3, W4)

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
(x_batch, t_batch) = preprocess(images)

test_loss = []
for ch in range(3):
    t_batch_tmp = t_batch[:, :, :,ch].reshape((size, 176, 176, 1)).transpose(0, 3, 1, 2).astype(np.float64) 
    x_batch_tmp = x_batch[:, :, :,ch].reshape((size, 44, 44, 1)).transpose(0, 3, 1, 2).astype(np.float64) / 255

    loss = test_model.loss(x_batch_tmp, t_batch_tmp)
    test_loss.append(loss)
with open('./weight/train/test_loss.pickle', 'wb') as f:
    pickle.dump(test_loss, f, pickle.HIGHEST_PROTOCOL)    


LR_images, HR_images, answer_images = test_model.test(images, size)

for i in range(size):
    img = cv2.cvtColor(LR_images[i], cv2.COLOR_BGR2RGB)
    plt.imshow(img) 
    plt.savefig('./images/output_images/LR/LR_' + str(i)+ '.png')

    img = cv2.cvtColor(HR_images[i], cv2.COLOR_BGR2RGB)
    plt.imshow(img) 
    plt.savefig('./images/output_images/HR/HR_' + str(i)+ '.png')


    img = cv2.cvtColor(answer_images[i], cv2.COLOR_BGR2RGB)
    plt.imshow(img) 
    plt.savefig('./images/output_images/ANSWER/ANSWER_' + str(i)+ '.png')
   

