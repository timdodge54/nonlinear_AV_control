import torch
from torch.utils.tensorboard import SummaryWriter
from enum import Enum


class Writer:
    def __init__(self, comment):
        self.writer = SummaryWriter(comment=comment)
        self.frame_ndx = 0
        
    def track_training(self, actor_loss, critic_loss, actuator_loss):
        self.writer.add_scalar("actor_loss", actor_loss, self.frame_idx)
        self.writer.add_scalar("critic_loss", critic_loss, self.frame_idx)
        self.writer.add_scalar("actuator_loss", actuator_loss, self.frame_idx)
        self.frame_ndx += 1
    def track_test(self, reward):
        self.writer.add_scalar("reward", reward, self.frame_idx)

def np_to_torch(x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.FloatTensor(x).to(device)

def torch_to_np(x):
    return x.cpu().detach().numpy()

