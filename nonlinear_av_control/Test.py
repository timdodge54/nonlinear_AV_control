from nonlinear_av_control.Env import BikeModelEnv
from nonlinear_av_control.Train import torch_to_numpy, numpy_to_torch
from nonlinear_av_control.Model import ActorCritic
import matplotlib.pyplot as plt
import torch
import os


class Hypers:
    n_envs = 1
    steps = 500
    dt = 1


def test(env, model):
    state = env.reset()
    test_reward = 0.0
    with torch.no_grad():
        for i in range(Hypers.steps):
            mu_action = model.actor(numpy_to_torch(state))
            state, rewards, done, _ = env.step(torch_to_numpy(mu_action))
            test_reward += rewards
            env.render()
            plt.cla()


if __name__ == "__main__":
    env = BikeModelEnv(Hypers.n_envs, Hypers.dt, Hypers.steps)
    in_, out_ = env.in_n_out
    model = ActorCritic(in_, out_, 256).to("cuda")
    model_name = os.path.join(os.path.dirname(__file__), "../models/model_copy.pt")
    model.load_model(model_name)
    for i in range(10):
        test(env, model)
