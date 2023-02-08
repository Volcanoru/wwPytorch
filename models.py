# -*- coding:utf-8 -*-
"""
@Time: 2022/03/01 22:23
@Author: KI
@File: models.py
@Motto: Hungry And Humble
"""
import numpy as np
from torch import nn


class BP:
    def __init__(self, args, file_name):
        self.file_name = file_name
        self.len = 0
        self.args = args
        self.input = np.zeros((args.B, args.input_dim))  # self.B samples per round
        self.w1 = 2 * np.random.random((args.input_dim, 50)) - 1  # limit to (-1, 1)
        self.z1 = 2 * np.random.random((args.B, 50)) - 1
        
        self.hidden_layer_1 = np.zeros((args.B, 50))
        self.w2 = 2 * np.random.random((50, 50)) - 1
        self.z2 = 2 * np.random.random((args.B, 50)) - 1
        
        self.hidden_layer_2 = np.zeros((args.B, 50))
        self.w3 = 2 * np.random.random((50, 50)) - 1
        self.z3 = 2 * np.random.random((args.B, 50)) - 1
        
        self.hidden_layer_3 = np.zeros((args.B, 50))
        self.w4 = 2 * np.random.random((50, 1)) - 1
        self.z4 = 2 * np.random.random((args.B, 1)) - 1
        
        self.output_layer = np.zeros((args.B, 1))
        self.loss = np.zeros((args.B, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deri(self, x):
        return x * (1 - x)

    def ReLu(self, x):
        return np.maximum(0,x)
    
    def ReLu_deri(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
    
    
    def forward_prop(self, data, label):
        self.input = data
        self.z1 = np.dot(self.input, self.w1)
        self.hidden_layer_1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.hidden_layer_1, self.w2)
        self.hidden_layer_2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.hidden_layer_2, self.w3)
        self.hidden_layer_3 = self.sigmoid(self.z3)
        self.z4 = np.dot(self.hidden_layer_3, self.w4)
        self.output_layer = self.sigmoid(self.z4)
        # error
        self.loss = 1 / 2 * (label - self.output_layer) ** 2

        return self.output_layer
    

    '''
    def forward_prop(self, data, label):
        self.input = data
        self.z1 = np.dot(self.input, self.w1)
        self.hidden_layer_1 = self.ReLu(self.z1)
        self.z2 = np.dot(self.hidden_layer_1, self.w2)
        self.hidden_layer_2 = self.ReLu(self.z2)
        self.z3 = np.dot(self.hidden_layer_2, self.w3)
        self.hidden_layer_3 = self.ReLu(self.z3)
        self.z4 = np.dot(self.hidden_layer_3, self.w4)
        self.output_layer = self.ReLu(self.z4)
        # error
        self.loss = 1 / 2 * (label - self.output_layer) ** 2

        return self.output_layer
    '''

    
    
    def backward_prop(self, label):
        # w4
        l_deri_out = self.output_layer - label
        l_deri_z4 = l_deri_out * self.sigmoid_deri(self.output_layer)
        l_deri_w4 = np.dot(self.hidden_layer_3.T, l_deri_z4)
        # w3
        l_deri_h3 = np.dot(l_deri_z4, self.w4.T)
        l_deri_z3 = l_deri_h3 * self.sigmoid_deri(self.hidden_layer_3)
        l_deri_w3 = np.dot(self.hidden_layer_2.T, l_deri_z3)
        # w2
        l_deri_h2 = np.dot(l_deri_z3, self.w3.T)
        l_deri_z2 = l_deri_h2 * self.sigmoid_deri(self.hidden_layer_2)
        l_deri_w2 = np.dot(self.hidden_layer_1.T, l_deri_z2)
        # w1
        l_deri_h1 = np.dot(l_deri_z2, self.w2.T)
        l_deri_z1 = l_deri_h1 * self.sigmoid_deri(self.hidden_layer_1)
        l_deri_w1 = np.dot(self.input.T, l_deri_z1)
        # update
        self.w4 -= self.args.lr * l_deri_w4
        self.w3 -= self.args.lr * l_deri_w3
        self.w2 -= self.args.lr * l_deri_w2
        self.w1 -= self.args.lr * l_deri_w1
    
    '''
    def backward_prop(self, label):
        # w4
        l_deri_out = self.output_layer - label
        l_deri_z4 = l_deri_out * self.ReLu_deri(self.output_layer)
        l_deri_w4 = np.dot(self.hidden_layer_3.T, l_deri_z4)
        # w3
        l_deri_h3 = np.dot(l_deri_z4, self.w4.T)
        l_deri_z3 = l_deri_h3 * self.ReLu_deri(self.hidden_layer_3)
        l_deri_w3 = np.dot(self.hidden_layer_2.T, l_deri_z3)
        # w2
        l_deri_h2 = np.dot(l_deri_z3, self.w3.T)
        l_deri_z2 = l_deri_h2 * self.ReLu_deri(self.hidden_layer_2)
        l_deri_w2 = np.dot(self.hidden_layer_1.T, l_deri_z2)
        # w1
        l_deri_h1 = np.dot(l_deri_z2, self.w2.T)
        l_deri_z1 = l_deri_h1 * self.ReLu_deri(self.hidden_layer_1)
        l_deri_w1 = np.dot(self.input.T, l_deri_z1)
        # update
        self.w4 -= self.args.lr * l_deri_w4
        self.w3 -= self.args.lr * l_deri_w3
        self.w2 -= self.args.lr * l_deri_w2
        self.w1 -= self.args.lr * l_deri_w1
    '''
