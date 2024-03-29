import torch
from torch import nn
from torch.nn import functional as F

class DQN(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(observation_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, action_dim)
        
    def forward(self, observation):
#         x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x