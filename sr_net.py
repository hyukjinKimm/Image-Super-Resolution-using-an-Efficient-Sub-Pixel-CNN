import numpy as np
from collections import OrderedDict
from common.function import *
from common.layer import *
from common.preprocess import *
from common.optimizer import *


class SR_Net:
    def __init__(self, W1 = 0.01*(-2*np.random.rand(64, 1 , 5, 5) + 1),
                       W2 = 0.01*(-2*np.random.rand(32, 64 , 3, 3) + 1), 
                       W3 = 0.01*(-2*np.random.rand(32, 32 , 3, 3) + 1),
                       W4 = 0.01*(-2*np.random.rand(16, 32 , 3, 3) + 1)):
        self.params = {}
        self.params['W1'] = W1
        self.params['W2'] = W2
        self.params['W3'] = W3
        self.params['W4'] = W4

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], np.zeros(shape=(64, 1, 1)), 1, 2)
        self.layers['Relu1'] = Relu() 
        
        self.layers['Conv2'] = Convolution(self.params['W2'], np.zeros(shape=(32, 1, 1)), 1, 1)
        self.layers['Relu2'] = Relu() 
        
        self.layers['Conv3'] = Convolution(self.params['W3'], np.zeros(shape=(32, 1, 1)), 1, 1)
        self.layers['Relu3'] = Relu() 
        
        self.layers['Conv4'] = Convolution(self.params['W4'], np.zeros(shape=(16, 1, 1)), 1, 1)
        self.layers['Relu4'] = Relu() 
        
        self.lastLayer = SubPixel(factor=4)
    
    # 입력 x 는 (B, 1, 44, 44) 저화질 이미지
    def predict(self, x):
        """추론을 수행"""
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    # 입력 x 는 (B, 1, 44, 44) 저화질 이미지
    # 입력 t 는 answer images  (B, 1, 176, 176)
    def loss(self, x, t):

        y = self.predict(x)
        z = self.lastLayer.forward(y)
        mse = MSE(z, t)

        return  mse


    def gradient(self, x, t):
        """오차역전파법으로 기울기를 구함"""
        # 순전파 
        self.loss(x, t)
 
        # 역전파
        dout = self.lastLayer.backward(t)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 결과 저장
        grads = {}
        for idx, key in enumerate(['W1', 'W2', 'W3', 'W4']):
            grads[key] = self.layers['Conv' + str(idx+1)].dW
        return grads

    def test(self, images, size): 
        LR_images, answer_images = preprocess(images)
        B = LR_images[:, :, :,0].reshape((size, 44, 44, 1)).transpose(0, 3, 1, 2).astype(np.float64) / 255
        G = LR_images[:, :, :,1].reshape((size, 44, 44, 1)).transpose(0, 3, 1, 2).astype(np.float64) / 255
        R = LR_images[:, :, :,2].reshape((size, 44, 44, 1)).transpose(0, 3, 1, 2).astype(np.float64) / 255
        
        R = self.predict(R)
        R = self.lastLayer.forward(R).transpose(0, 2, 3, 1)

        G = self.predict(G)
        G = self.lastLayer.forward(G).transpose(0, 2, 3, 1)
    
        B = self.predict(B)
        B = self.lastLayer.forward(B).transpose(0, 2, 3, 1)
        HR_images = np.zeros((size, 176, 176, 3))

        for idx in range(size):
            HR_images[idx] = merged_img = cv2.merge((B[idx].reshape((176, 176)),G[idx].reshape((176, 176)),R[idx].reshape((176, 176))))

        # LR_images = (B, 44, 44, 3)
        # HR_imgaes = (B, 176, 176, 3) B G R 순서
        # answer_images = (B, 176, 176, 3)
        return LR_images.astype(np.uint8), HR_images.astype(np.uint8), answer_images.astype(np.uint8)

    def test2(self, images, size, model): 
        LR_images, answer_images = preprocess(images)

        B = LR_images[:, :, :,0].reshape((size, 44, 44, 1)).transpose(0, 3, 1, 2).astype(np.float64) / 255
        G = LR_images[:, :, :,1].reshape((size, 44, 44, 1)).transpose(0, 3, 1, 2).astype(np.float64) / 255
        R = LR_images[:, :, :,2].reshape((size, 44, 44, 1)).transpose(0, 3, 1, 2).astype(np.float64) / 255
        
        R = model['R'].predict(R)
        R = model['R'].lastLayer.forward(R).transpose(0, 2, 3, 1)

        G = model['G'].predict(G)
        G = model['G'].lastLayer.forward(G).transpose(0, 2, 3, 1)
    
        B = model['B'].predict(B)
        B = model['B'].lastLayer.forward(B).transpose(0, 2, 3, 1)

        HR_images = np.zeros((size, 176, 176, 3))

        for idx in range(size):
            HR_images[idx] =  cv2.merge((B[idx].reshape((176, 176)),G[idx].reshape((176, 176)),R[idx].reshape((176, 176))))

        # LR_images = (B, 44, 44, 3)
        # HR_imgaes = (B, 176, 176, 3) B G R 순서
        # answer_images = (B, 176, 176, 3)
        return LR_images.astype(np.uint8), HR_images.astype(np.uint8), answer_images.astype(np.uint8)
    
