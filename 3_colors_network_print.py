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

font1 = {'family': 'serif',
      'color':  'darkred',
      'weight': 'bold',
      'size': 10}
font2 = {'family': 'Times New Roman',
      'color':  'blue',
      'weight': 'bold',
      'size': 10}

font3 = {'family': 'Arial',
      'color':  'forestgreen',
      'weight': 'bold',
      'style': 'italic',
      'size': 10}

train_loss = {}
with open("./weight/color_weight/index.pickle","rb") as fr:
        idx = pickle.load(fr)

for key in ['B', 'G', 'R']:
    with open("./weight/color_weight/" + key +"/train_loss.pickle","rb") as fr:
        loss = pickle.load(fr)
    train_loss[key] = loss

X = np.array([i for i in range(50, idx, 30)])
Y = np.array([train_loss['B'][i] for i in range(50, idx, 30)])
plt.plot (X , Y , '.',label="B network", color='b')
plt.title('MSE of B')
plt.ylabel("MSE")
plt.xlabel("iters_num")
plt.axhline(y=np.min(Y), color='b', linewidth=1)
plt.text(0, np.min(Y) - 1.25, '%.1f' %np.min(Y), fontdict=font2)
plt.legend()
plt.savefig('./conclusion/3_colors_network/loss_B.png')
plt.show()

X = np.array([i for i in range(50, idx, 30)])
Y = np.array([train_loss['G'][i] for i in range(50, idx, 30)])
plt.plot (X , Y , '.',label="G network", color='g')
plt.title('MSE of G')
plt.ylabel("MSE")
plt.xlabel("iters_num")
plt.axhline(y=np.min(Y), color='g', linewidth=1)
plt.text(0, np.min(Y) - 1.25, '%.1f' %np.min(Y), fontdict=font3)
plt.legend()
plt.savefig('./conclusion/3_colors_network/loss_G.png')
plt.show()

X = np.array([i for i in range(50, idx, 30)])
Y = np.array([train_loss['R'][i] for i in range(50, idx, 30)])
plt.plot (X , Y , '.',label="R network", color='r')
plt.title('MSE of R')
plt.ylabel("MSE")
plt.xlabel("iters_num")
plt.axhline(y=np.min(Y), color='r', linewidth=1)
plt.text(0, np.min(Y) - 1.25, '%.1f' %np.min(Y), fontdict=font1)
plt.legend()
plt.savefig('./conclusion/3_colors_network/loss_R.png')
plt.show()

X = np.array([i for i in range(50, idx, 30)])
Y = np.array([train_loss['B'][i] for i in range(50, idx, 30)])
plt.plot (X , Y , '.',label="B network", color='b')
plt.axhline(y=np.min(Y), color='b', linewidth=1)
plt.text(0, np.min(Y) +10, '%.1f' %np.min(Y), fontdict=font2)

X = np.array([i for i in range(50, idx, 30)])
Y = np.array([train_loss['G'][i] for i in range(50, idx, 30)])
plt.plot (X , Y , '.',label="G network", color='g')
plt.axhline(y=np.min(Y), color='g', linewidth=1)
plt.text(0, np.min(Y) -20, '%.1f' %np.min(Y), fontdict=font3)

X = np.array([i for i in range(50, idx, 30)])
Y = np.array([train_loss['R'][i] for i in range(50, idx, 30)])
plt.plot (X , Y , '.',label="R network", color='r')
plt.axhline(y=np.min(Y), color='r', linewidth=1)
plt.text(0, np.min(Y) -50, '%.1f' %np.min(Y), fontdict=font1)
plt.title('MSE of RGB')
plt.ylabel("MSE")
plt.xlabel("iters_num")
plt.legend()
plt.savefig('./conclusion/3_colors_network/loss_RGB.png')
plt.show()


src = './images/3_colors_network_images/LR/LR_7.png'
dir = './conclusion/3_colors_network/LR.png'
shutil.copy(src, dir)
src = './images/3_colors_network_images/HR/HR_7.png'
dir = './conclusion/3_colors_network/HR.png'
shutil.copy(src, dir)
src = './images/3_colors_network_images/ANSWER/ANSWER_7.png'
dir = './conclusion/hard_train/ANSWER.png'
shutil.copy(src, dir)


fig = plt.figure()
rows = 1
cols = 3
img1 = cv2.imread('./images/3_colors_network_images/LR/LR_7.png')
img2 = cv2.imread('./images/3_colors_network_images/HR/HR_7.png')
img3 = cv2.imread('./images/3_colors_network_images/ANSWER/ANSWER_7.png')
ax1 = fig.add_subplot(rows, cols, 1)
ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
ax1.set_title('LR')
ax1.axis("off")
ax2 = fig.add_subplot(rows, cols, 2)
ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
ax2.set_title('HR')
ax2.axis("off")
ax3 = fig.add_subplot(rows, cols, 3)
ax3.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
ax3.set_title('ANSWER')
ax3.axis("off")

plt.savefig('./conclusion/3_colors_network/output.png')
