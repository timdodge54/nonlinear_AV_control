import torch
from torch.utils.tensorboard import SummaryWriter


class Writer:
    def __init__(self, comment):
        self.writer = SummaryWriter(comment=comment)
        self.frame_idx = 0
        
    def track_training(self, actor_loss, critic_loss, actuator_loss, distance_to_waypoint):
        self.writer.add_scalar("loss/actor_loss", actor_loss, self.frame_idx)
        self.writer.add_scalar("loss/critic_loss", critic_loss, self.frame_idx)
        self.writer.add_scalar("loss/actuator_loss", actuator_loss, self.frame_idx)
        self.writer.add_scalar("distance_to_waypoint", distance_to_waypoint, self.frame_idx)
        self.frame_idx += 1
    def track_test(self, reward):
        self.writer.add_scalar("reward", reward, self.frame_idx)

def np_to_torch(x):
    device = torch.device("cuda")
    return torch.FloatTensor(x).to(device)

def torch_to_np(x):
    return x.cpu().detach().numpy()

