import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.nn.modules import loss


def get_activation_fn(activation_name='selu'):
    if activation_name is None:
        return None
    elif activation_name == 'selu':
        return torch.nn.SELU
    elif activation_name == 'softmax':
        return torch.nn.Softmax
    elif activation_name == 'tanh':
        return torch.nn.Tanh
    elif activation_name == 'elu':
        return torch.nn.ELU
    elif activation_name == 'relu':
        return torch.nn.ReLU
    elif activation_name == 'sigmoid':
        return torch.nn.Sigmoid
    elif activation_name == 'linear':
        return torch.nn.Linear
    elif activation_name == 'softplus':
        return torch.nn.Softplus
    elif activation_name == 'softsign':
        return torch.nn.Softsign
    elif activation_name == 'leaky_relu':
        return torch.nn.LeakyReLU
    else:
        return torch.nn.Tanh


def get_optimizer(optimizer_name='adam'):
    if optimizer_name is None:
        return None
    elif optimizer_name == 'adam':
        return torch.optim.Adam
    elif optimizer_name == 'rmsprop':
        return torch.optim.RMSprop
    elif optimizer_name == 'adadelta':
        return torch.optim.Adadelta
    elif optimizer_name == 'adagrad':
        return torch.optim.Adagrad
    elif optimizer_name == 'sgd':
        return torch.optim.SGD
    elif optimizer_name == 'asgd':
        return torch.optim.ASGD
    elif optimizer_name == 'lbfgs':
        return torch.optim.LBFGS
    else:
        return torch.optim.Adam


def get_loss_func(loss_func_name='smoothl1'):
    if loss_func_name is None:
        return None
    if loss_func_name == 'smoothl1':
        return loss.SmoothL1Loss()
    elif loss_func_name == 'mae':
        return loss.L1Loss()
    elif loss_func_name == 'mse':
        return loss.MSELoss()
    elif loss_func_name == 'cosine':
        return loss.CosineEmbeddingLoss()
    elif loss_func_name == 'hinge':
        return loss.HingeEmbeddingLoss()
    else:
        return loss.MSELoss()


def get_true_reward(reward_from_unity):
    # real_reward = reward_from_unity - 10000.0 if reward_from_unity > 5000 else reward_from_unity
    if reward_from_unity > 15000:
        real_reward = reward_from_unity - 20000.0
    elif reward_from_unity > 5000:
        real_reward = reward_from_unity - 10000.0
    else :
        real_reward = reward_from_unity

    if real_reward > 100.0:
        reward = -real_reward + 90.0
    elif real_reward > 0.01:
        reward = -real_reward + 4.0
    else:
        reward = -10.0

    return reward