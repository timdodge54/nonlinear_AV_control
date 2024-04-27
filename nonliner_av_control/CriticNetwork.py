import os

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ament_index_python.packages import get_package_share_directory


class CriticNetwork(nn.Module):
    def __init__(
        self,
        beta,
        input_dims,
        fc1_dims,
        fc2_dims,
        fc3_dims,
        n_actions,
        name,
        chkpt_dir="Models/ddpg",
    ) -> None:
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions
        rel_path = get_package_share_directory("reinforcement_planning")
        rel_path = os.path.join(rel_path, "ddpg_planning")
        placehoder_dir = os.path.join(chkpt_dir, name + "_ddpg")
        self.checkpoint_file = os.path.join(rel_path, placehoder_dir)

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        f3 = 1 / np.sqrt(self.fc3.weight.data.size()[0])
        T.nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        T.nn.init.uniform_(self.fc3.bias.data, -f3, f3)
        self.bn3 = nn.LayerNorm(self.fc3_dims)

        self.action_value = nn.Linear(self.n_actions, fc3_dims)
        f4 = 0.003
        self.q = nn.Linear(self.fc3_dims, 1)
        T.nn.init.uniform_(self.q.weight.data, -f4, f4)
        T.nn.init.uniform_(self.q.bias.data, -f4, f4)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device("cpu")  # "cuda:0" if T.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc3(state_value)
        state_value = self.bn3(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print("...saving checkpoint...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("...loading checkpoint...")
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))