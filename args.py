# -*- coding:utf-8 -*-
"""
@Time: 2022/03/01 11:52
@Author: KI
@File: args.py
@Motto: Hungry And Humble
"""
import argparse # 设置参数的一个python包
import torch


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--E', type=int, default=50, help='number of rounds of training')
    parser.add_argument('--r', type=int, default=10, help='number of communication rounds')
    parser.add_argument('--K', type=int, default=11, help='number of total clients')
    parser.add_argument('--input_dim', type=int, default=4, help='input dimension') #这里要根据网络结构调整
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--C', type=float, default=1, help='sampling rate')
    parser.add_argument('--B', type=int, default=100, help='local batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    clients = ['S' + str(i) for i in range(0, 20)]
    parser.add_argument('--clients', default=clients)

    args = parser.parse_args()

    return args

