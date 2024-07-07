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
            state = env.reset()
            x_history = []
            y_history = []
            reward_history = []
            steps = []
            for i in range(Hypers.steps):
                mu_action = model.actor(np_to_torch(state))
                state, reward, _, _ = env.step(torch_to_np(mu_action))
                test_reward += torch.mean(reward)
                x_history.append(float(state.item(0)))
                y_history.append(float(state.item(1)))
                reward_history.append(reward.cpu().numpy())
                steps.append(i)
            index = 1400
            xr = env.xd[index]
            yr = env.yd[index]
            # 2 plots (x, steps) and (y, steps)
            fig, ax = plt.subplots(3)
            ax[0].plot(steps, x_history)
            ax[0].axhline(y=xr, color='r', linestyle='--')
            ax[0].set_title('X vs Steps')
            ax[0].set_xlabel('Steps')
            ax[0].set_ylabel('X')
            ax[1].plot(steps, y_history)
            ax[1].axhline(y=yr, color='r', linestyle='--')
            ax[1].set_title('Y vs Steps')
            ax[1].set_xlabel('Steps')
            ax[1].set_ylabel('Y')
            ax[2].plot(steps, reward_history)
            ax[2].set_title('Reward vs Steps')
            ax[2].set_xlabel('Steps')
            ax[2].set_ylabel('Reward')
            plt.show()
            
            
            

if __name__ == "__main__":
    test()