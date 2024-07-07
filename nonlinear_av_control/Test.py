import torch
from nonlinear_av_control.Env import Env
from nonlinear_av_control.Model import ActorCritic
from nonlinear_av_control.Utils import np_to_torch, torch_to_np
import matplotlib.pyplot as plt


class Hypers:
    n_envs = 1
    std_dev = 0.2
    dt = .01
    steps = 6414
    n_test_loops = 10


def test():
    env = Env(n=Hypers.n_envs, dt=Hypers.dt, max_timesteps=Hypers.steps)
    in_, out = env.n_in_out
    model = ActorCritic(in_, out).to('cuda')
    model.load()
    state = env.reset()
    with torch.no_grad():
        for _ in range(Hypers.n_test_loops):
            test_reward = 0
            i = 0
            state = env.reset()
            for i in range(Hypers.steps):
                mu_action = model.actor(np_to_torch(state))
                state, reward, _, _ = env.step(torch_to_np(mu_action))
                test_reward += torch.mean(reward)
                if i % 10 == 0:
                    print(f"Step {i} Reward: {test_reward}")
                    print(f"state: {state}")
                    print(f"action: {mu_action}")
                env.render()

if __name__ == "__main__":
    test()
    plt.show()