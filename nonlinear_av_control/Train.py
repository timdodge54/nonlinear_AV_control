import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

from nonlinear_av_control.Model import ActorCritic

from nonlinear_av_control.Utils import np_to_torch, torch_to_np, Writer
from nonlinear_av_control.Env import Env
from itertools import count


class Hypers:
    n_envs = 124
    learning_rate = 1e-4
    std_dev = 0.2
    critic_discount = 0.5
    ppo_eps = 0.2
    ppo_epochs= 10
    gamma = 0.99
    dt = .01
    steps = 6414
    gae_lambda = 0.95
    
class TrajectoryData:
    def __init__(self, num_inputs, num_outputs, num_envs, n_steps) -> None:
        self.s = n_steps
        i, o, n, s = num_inputs, num_outputs, num_envs, n_steps
        torch.set_default_device('cuda')
        self.states = torch.zeros((n, s, i))
        self.values = torch.zeros((n, s + 1, 1))
        self.actions = torch.zeros((n, s, o))
        self.log_probs = torch.zeros((n, s, o))
        self.rewards = torch.zeros((n, s, 1 ))
        self.masks = torch.ones((n, s, 1 ))
        self.returns = torch.zeros((n, s, 1 ))
        self.advantages = torch.zeros((n, s, 1))
        self.as_dict = vars(self)

    def store_step(self, i, keys = ["log_probs", "states", "actions", "rewards", "masks", "values"], vals = None):
        for key, val in zip(keys, vals):
            self.as_dict[key][:, i, :] = val

    def calc_gae_returns_advantage(self):
        gae = torch.zeros(Hypers.n_envs, 1).to('cuda')
        for i in range(Hypers.steps -1, 0, -1):
            delta = (
                self.rewards[:, i] + Hypers.gamma * self.values[:, i+1] * self.masks[:, i] - self.values[:, i]
            )
            gae = delta + Hypers.gamma * Hypers.gae_lambda * self.masks[:, i] * gae
            self.returns[:, i] = gae + self.values[:, i]
        advantage = self.rewards - self.values[:, :-1]
        self.advantages = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

class PPO:
    def __init__(self):
        self.env = Env(Hypers.n_envs, Hypers.dt, Hypers.steps) 
        num_inputs, num_outputs = self.env.n_in_out
        self.model = ActorCritic(num_inputs, num_outputs).to('cuda')
        self.optim = torch.optim.Adam(self.model.parameters(), lr=Hypers.learning_rate)
        self.traj_data = TrajectoryData(num_inputs, num_outputs, Hypers.n_envs, Hypers.steps)
        self.writer = Writer(f"ppo_{Hypers.n_envs}_{Hypers.learning_rate}_{Hypers.std_dev}")

    def collect_rollouts(self):
        state = np_to_torch(self.env.reset())
        with torch.no_grad():
            for i in range(Hypers.steps):
                value = torch.mean(self.model.critic(state), axis=1, keepdim=True)
                dist = Normal(self.model.actor(state), Hypers.std_dev)
                action = dist.sample()
                next_state, reward, done, _ = self.env.step(torch_to_np(action))
                self.traj_data.store_step(i, vals=[dist.log_prob(action), state, action, reward, 1-done, value])    
                state = np_to_torch(next_state)
                if done:
                    break
        # collect and store final value estimate
        self.traj_data.values[:, -1] = torch.mean(self.model.critic(state), axis=1, keepdim=True)
        self.traj_data.calc_gae_returns_advantage()

    def update_model(self):
        actor_losses, critic_losses, actuation_losses = [], [], []

        for _ in range(Hypers.ppo_epochs):

            # Compute values and critic loss
            values = torch.mean(self.model.critic(self.traj_data.states), axis=-1, keepdim=True)
            critic_loss = ((self.traj_data.returns.clone().detach() - values).pow(2)).mean()

            # Compute mu_action and log_probs
            mu_action = self.model.actor(self.traj_data.states)
            dist = Normal(mu_action, Hypers.std_dev)
            log_probs = dist.log_prob(self.traj_data.actions)
            ratio = (log_probs - self.traj_data.log_probs).exp()

            # Compute actor loss
            unclipped = ratio * self.traj_data.advantages.clone().detach()
            clipped = torch.clamp(ratio, 1 - Hypers.ppo_eps, 1 + Hypers.ppo_eps) * self.traj_data.advantages.clone().detach()
            actor_loss = -torch.min(unclipped, clipped).mean()

            # Compute actuation loss
            actuation_loss = torch.mean(torch.clamp(mu_action.pow(2) - 1, 0, torch.inf))*0

            # Aggregate the losses
            self.optim.zero_grad()
            (actor_loss + Hypers.critic_discount * critic_loss + actuation_loss).backward()

            self.optim.step()
            self.optim.zero_grad()

            # Collect the losses for tracking
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            actuation_losses.append(actuation_loss.item())

        # Track the training losses after the loop
        self.writer.track_training(
            np.mean(actor_losses),
            np.mean(critic_losses),
            np.mean(actuation_losses),
            np.mean(self.env.distance_to_waypoint)
        )

    def test(self):
        state = self.env.reset()
        test_reward = 0
        with torch.no_grad():
            for _ in range(Hypers.steps):
                mu_action = self.model.actor(np_to_torch(state))
                state, reward, _, _ = self.env.step(torch_to_np(mu_action))
                test_reward += torch.mean(reward)
        self.writer.track_test(test_reward/Hypers.steps)
    
    def train(self):    
        print("Training")
        for i in count(0):
            self.collect_rollouts()
            self.update_model()
            if i % 10 == 0:
                print(f"Step {i}")
                self.test()
                self.model.save()

if __name__ == "__main__":
    ppo = PPO()
    ppo.train()