import torch
from torch import nn

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(ActorCritic, self).__init__()
        self.hidden_size = hidden_size
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def save(self, model_name=None):
        if model_name is None:
            torch.save(self.state_dict(), 'model.pth')
        else:
            torch.save(self.state_dict(), model_name)
    def load(self, model_name=None):
        if model_name is None:
            self.load_state_dict(torch.load('model.pth'))
        else:
            self.load_state_dict(torch.load(model_name))