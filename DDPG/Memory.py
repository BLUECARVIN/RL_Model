import random
from collections import deque
import numpy as np


class MemoryBuffer:# Continous env memory
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.len = 0
    
    def sample(self, count):
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)
        
        s_arr = np.float32([arr[0] for arr in batch])
        a_arr = np.float32([arr[1] for arr in batch])  # action id
        r_arr = np.float32([arr[2] for arr in batch])
        s1_arr = np.float32([arr[3] for arr in batch])
        done = np.array([arr[4] for arr in batch])
        
        return s_arr, a_arr, r_arr, s1_arr, done
    
    def len(self):
        return self.len
    
    def add(self, s, a, r, s1, done):
        transition = (s, a, r, s1, done)
        self.len += 1
        if self.len>self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)