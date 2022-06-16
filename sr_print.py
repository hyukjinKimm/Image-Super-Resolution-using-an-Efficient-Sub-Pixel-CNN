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


src = './images/output_images/LR/LR_7.png'
dir = './conclusion/hard_train/LR.png'
shutil.copy(src, dir)
src = './images/output_images/HR/HR_7.png'
dir = './conclusion/hard_train/HR.png'
shutil.copy(src, dir)
src = './images/output_images/ANSWER/ANSWER_7.png'
dir = './conclusion/hard_train/ANSWER.png'
shutil.copy(src, dir)

fig = plt.figure()
rows = 1
cols = 3
img1 = cv2.imread('./images/output_images/LR' + '/LR_7.png')
img2 = cv2.imread('./images/output_images/HR' + '/HR_7.png')
img3 = cv2.imread('./images/output_images/ANSWER' + '/ANSWER_7.png')
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
plt.show()
plt.savefig('./conclusion/hard_train/output.png')
