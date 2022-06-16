import numpy as np
import torch
from common.function import *

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 중간 데이터（backward 시 사용）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 가중치와 편향 매개변수의 기울기
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
       
        out = np.dot(col, col_W)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
class SubPixel:
    def __init__(self, factor=4):
        self.factor = factor

        self.input = None #Tensor 형 type 이 들어감
        self.output = None #numpy 타입의 고화질 이미지

    def forward(self, input):
        # 만들어낸 고화질 img 를 return 함
        self.input = torch.Tensor(input)
        shape = self.input.shape
    
        if self.input.ndim == 3:  
            self.input = self.input.reshape(1, shape[0], shape[1], shape[2])
            shape = self.input.shape

        B, C, H, W = shape 
        out__C = C / self.factor**2

        # output = (B, out_C, N, N)
        pixel_shuffle = torch.nn.PixelShuffle(self.factor)
        self.output = pixel_shuffle(self.input).numpy()

        return self.output

    def backward(self, answer):
        B, C, H, W = self.output.shape
        pixel_Unshuffle = torch.nn.PixelUnshuffle(self.factor)
       
        answer = pixel_Unshuffle(torch.Tensor(answer))
       
        dout = 2*(self.input.numpy() - answer.numpy()) / (B * C * H * W)
        
        return dout

