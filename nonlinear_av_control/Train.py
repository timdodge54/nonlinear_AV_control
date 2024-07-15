import numpy as np
import torch
from nonlinear_av_control.Env import BikeModelEnv
from nonlinear_av_control.Model import ActorCritic
from torch.distributions import Normal
from itertools import count
from torch.utils.tensorboard import SummaryWriter
import os


def torch_to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def numpy_to_torch(array):
    return torch.from_numpy(array).float().to("cuda")


class Writer:
    def __init__(self):
        self.writer = SummaryWriter()
        self.frame_idx = 0

    def track_training(self, actor_loss, critic_loss):
        self.writer.add_scalar("actor_loss", actor_loss, self.frame_idx)
        self.writer.add_scalar("critic_loss", critic_loss, self.frame_idx)
        self.frame_idx += 1

    def track_rewards(self, rewards):
        self.writer.add_scalar("rewards", rewards, self.frame_idx)


class Hypers:
    n_envs = 1024
    learning_rate = 1e-4
    std_dev = 0.2
    steps = 500
    ppo_eps = 0.1
    ppo_epochs = 10
    gamma = 0.99
    gae_lambda = 0.95


class TrajectoryData:
    def __init__(self, n_inputs, n_outputs, n_steps, n_envs):
        self.n_steps = n_steps
        self.n_envs = n_envs
        i, o, n, s = n_inputs, n_outputs, n_envs, n_steps
        torch.set_default_device("cuda")
        self.states = torch.zeros((n, s, i))
        self.values = torch.zeros((n, s + 1, 1))
        self.actions = torch.zeros((n, s, o))
        self.log_probs = torch.zeros((n, s, o))
        self.rewards = torch.zeros((n, s, 1))
        self.not_done_masks = torch.zeros((n, s, 1))
        self.returns = torch.zeros((n, s, 1))
        self.advantages = torch.zeros((n, s, 1))
        self.as_dict = vars(self)

    def store_step(
        self,
        step,
        keys=["states", "values", "actions", "log_probs", "rewards", "not_done_masks"],
        vals=None,
    ):
        for key, val in zip(keys, vals):
            self.as_dict[key][:, step] = val.clone()

    def calc_gae_returns_advantage(self):
        gae = torch.zeros((self.n_envs, 1)).to("cuda")
        for i in range(self.n_steps - 1, -1, -1):  # Corrected range to include 0
            delta = (
                self.rewards[:, i]
                + Hypers.gamma * self.values[:, i + 1] * self.not_done_masks[:, i]
                - self.values[:, i]
            )
            gae = (
                delta
                + Hypers.gamma * Hypers.gae_lambda * self.not_done_masks[:, i] * gae
            )
            self.returns[:, i] = gae + self.values[:, i]
        advantage = self.returns - self.values[:, :-1]
        self.advantages = (advantage - advantage.mean()) / (advantage.std() + 1e-8)


class PPO:
    def __init__(self):
        self.env = BikeModelEnv(Hypers.n_envs, 0.1, Hypers.steps)
        in_, out_ = self.env.in_n_out
        self.model = ActorCritic(in_, out_, 256).to("cuda")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=Hypers.learning_rate
        )
        self.t_data = TrajectoryData(in_, out_, Hypers.steps, Hypers.n_envs)
        self.writer = Writer()

    def collect_rollouts(self):
        state = numpy_to_torch(self.env.reset())
        with torch.no_grad():
            for i in range(Hypers.steps):
                values = self.model.critic_forward(state)
                dist = Normal(self.model.actor_forward(state), Hypers.std_dev)
                action = dist.sample()
                next_state, reward, done, _ = self.env.step(torch_to_numpy(action))
                self.t_data.store_step(
                    i,
                    vals=[
                        state,
                        values,
                        action,
                        dist.log_prob(action),
                        numpy_to_torch(reward).unsqueeze(1),
                        numpy_to_torch(1 - done).unsqueeze(1),
                    ],
                )
                state = numpy_to_torch(next_state)

            self.t_data.values[:, -1] = self.model.critic_forward(state)
            self.t_data.calc_gae_returns_advantage()

    def update_policy(self):
        for _ in range(Hypers.ppo_epochs):
            values = self.model.critic(self.t_data.states)
            critic_loss = (self.t_data.returns - values).pow(2).mean()
            mu_action = self.model.actor(self.t_data.states)
            dist = Normal(mu_action, Hypers.std_dev)
            log_probs = dist.log_prob(self.t_data.actions)
            ratio = (log_probs - self.t_data.log_probs).exp()
            unclipped = ratio * self.t_data.advantages
            clipped = (
                torch.clamp(ratio, 1 - Hypers.ppo_eps, 1 + Hypers.ppo_eps)
                * self.t_data.advantages
            )
            actor_loss = -torch.min(unclipped, clipped).mean()

            self.optimizer.zero_grad()
            loss = actor_loss + critic_loss
            loss.backward()
            self.optimizer.step()

        self.writer.track_training(actor_loss.item(), critic_loss.item())

    def test(self):
        state = self.env.reset()
        test_reward = 0.0
        with torch.no_grad():
            for i in range(Hypers.steps):
                mu_action = self.model.actor(numpy_to_torch(state))
                state, rewards, done, _ = self.env.step(torch_to_numpy(mu_action))
                test_reward += np.mean(rewards)
            self.writer.track_rewards(test_reward)

    def train(self):
        for i in count(0):
            self.collect_rollouts()
            self.update_policy()
            if i % 10 == 0:
                self.test()
                model_name = os.path.join(
                    os.path.dirname(__file__), "../models/model.pt"
                )
                self.model.save_model(model_name)
                print(f"Episode {i} complete...")


def train():
    torch.autograd.set_detect_anomaly(True)
    ppo = PPO()
    model_name = os.path.join(os.path.dirname(__file__), "../models/model_copy.pt")
    ppo.model.load_model(model_name)
    ppo.train()


if __name__ == "__main__":
    train()
