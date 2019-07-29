import numpy as np
import torch


def hard_update(target, source):
    """
    Copies the parameters from source network to target network
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
        

def soft_update(target, source, tau):
    """
    update the parameters from source network to target network
    y = /tau * x + (1 - /tau)* y
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)