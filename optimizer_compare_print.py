import os, cv2
import pickle
import numpy as np
from collections import OrderedDict
from common.function import *
from common.layer import *
from common.preprocess import *
from common.optimizer import *
import sr_net
import shutil

train_loss = {}
for key in ['SGD', 'Adam', 'AdaGrad', 'Momentum']:
    with open("./weight/optimizer_compare/" + key +"/train_loss.pickle","rb") as fr:
        loss = pickle.load(fr)
    train_loss[key] = loss

X = [i for i in range(3, 1000)]
Y = [sum(train_loss['Adam'][i:i+3])/3 for i in range(3, 1000)]
plt.plot (X, Y, label="Adam", color='crimson')
X = [i for i in range(3, 1000)]
Y = [sum(train_loss['AdaGrad'][i:i+3])/3 for i in range(3, 1000)]
plt.plot (X, Y,label="AdaGrad", color = 'magenta')
plt.legend() # 꼭 호출해 주어야만 legend가 달립니다
plt.title('Adam vs AdaGrad')
plt.savefig('./conclusion/optimizer_compare/Adam_AdaGrad.png')
plt.show()

X = [i for i in range(3, 1000)]
Y = [sum(train_loss['SGD'][i:i+3])/3 for i in range(3, 1000)]
plt.plot (X , Y, label="SGD", color = 'k')
X = [i for i in range(3, 1000)]
Y = [sum(train_loss['Momentum'][i:i+3])/3 for i in range(3, 1000)]
plt.plot (X, Y,label="Momentum", color = 'sienna')
plt.legend() # 꼭 호출해 주어야만 legend가 달립니다
plt.title('SGD vs Momentum')
plt.savefig('./conclusion/optimizer_compare/SGD_Momentum.png')
plt.show()

fig = plt.figure()
rows = 2
cols = 3
i = 0
for key in ['Adam', 'AdaGrad']:
    src = './images/optimizer_compare_images/' + key+ '/HR_7.png'
    dir = './conclusion/optimizer_compare/'+ key +'_HR.png'
    shutil.copy(src, dir)
    img1 = cv2.imread('./images/optimizer_compare_images/LR/LR_7.png')
    img2 = cv2.imread('./images/optimizer_compare_images/' + key + '/HR_7.png')
    img3 = cv2.imread('./images/optimizer_compare_images/ANSWER/ANSWER_7.png')
    i += 1
    ax1 = fig.add_subplot(rows, cols, i)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax1.set_title('LR')
    ax1.axis("off")
    i += 1
    ax2 = fig.add_subplot(rows, cols, i)
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax2.set_title('HR')
    ax2.axis("off")
    i += 1
    ax3 = fig.add_subplot(rows, cols, i)
    ax3.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    ax3.set_title('ANSWER')
    ax3.axis("off")
plt.savefig('./conclusion/optimizer_compare/Adam vs AdaGrad.png')
plt.show()

src = './images/optimizer_compare_images/LR/LR_7.png'
dir = './conclusion/optimizer_compare/LR.png'
shutil.copy(src, dir)

src = './images/optimizer_compare_images/ANSWER/ANSWER_7.png'
dir = './conclusion/optimizer_compare/ANSWER.png'
shutil.copy(src, dir)