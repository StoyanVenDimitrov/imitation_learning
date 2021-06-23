import time
import gym
import numpy as np
import torch as torch
import torch.utils.data as torch_data
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from generator import Generator
from discriminator import  Discriminator

class GAIL:
    """Class for training the GAIL algorithm 
    """

    def __init__(self, env_name) -> None:
        # self.env = gym.make(env_name)
        self.env = gym.make(env_name)
        self.env.seed(543)
        self.discriminator = Discriminator(state_shape=self.env.observation_space.shape[0], action_shape=self.env.action_space.n)
        self.generator = Generator(self.env, self.env.observation_space.shape[0], self.env.action_space.n, self.discriminator)

    def train_expert(self):
        """train a model to generate expert demons with it
        """
        model = PPO(
            "MlpPolicy", 
            self.env, 
            n_epochs=4,
            n_steps=256,
            verbose=1)
        model.learn(total_timesteps=100000)
        model.save(self.env.spec.id)
        return model

    def get_demonstrations(self):
        """With the PPO expert policy, create expert demonstrations
        """
        # env = gym.make(env_name)
        env_name = self.env.spec.id
        try:
            model = PPO.load(env_name, env=self.env)
        except FileNotFoundError:
            model = self.train_expert()
        obs = self.env.reset()
        flat_trajectories = {key: [] for key in ["state", "next_state", "action", "done"]}
        for i in range(100):
            flat_trajectories
            flat_trajectories["state"].append(obs)
            action, _states = model.predict(obs, deterministic=True)
            flat_trajectories["action"].append(action)
            obs, reward, done, info = self.env.step(action)
            flat_trajectories["next_state"].append(obs)
            flat_trajectories["done"].append(done)

            self.env.render()
            if done:
                obs = self.env.reset()
        assert np.array_equal(flat_trajectories["state"][1], flat_trajectories["next_state"][0])
        assert np.array_equal(flat_trajectories["state"][2], flat_trajectories["next_state"][1])
        assert np.array_equal(flat_trajectories["state"][-1], flat_trajectories["next_state"][-2])
        expert_dataset = ExpertDataset(flat_trajectories)
        expert_data_loader = torch_data.DataLoader(
                expert_dataset,
                batch_size=32,
                shuffle=True,
                drop_last=True,
            )
        # print(next(iter(expert_data_loader)))
        return expert_data_loader
        
    def train(self, exp_demos=None):
        """train alternating the discriminator and the generator

        Args:
            exp_demos ([type]): expert trajectories 
        """
        exp_dataloader = self.get_demonstrations()
        for i, data in enumerate(exp_dataloader, 0):
            disc_loss = self.discriminator.train(data, None)
            if i % 10 == 0:
                print(f'Batch {i}\tLast loss: {disc_loss}')



        #self.generator.train(200, 10000)
    
    def generate(self, n_rollouts):
        """generates trajectories with the model on the self.env

        Args:
            n_rollouts (int): number of gen trajectories
        """


    def rollout_stats(self, trajectories):
        """statistics for a set of imitation/expert trajectories

        Args:
            trajectories (dict): trajectories
        """
        pass


class ExpertDataset(torch_data.Dataset):
    def __init__(self, flat_trajectories):
        self.state = flat_trajectories["state"]
        self.next_state = flat_trajectories["next_state"]
        self.action = flat_trajectories["action"]
        self.done = flat_trajectories["done"]
        
    def __getitem__(self, index):
        return {
            "state": self.state[index], 
            "next_state": self.next_state[index],
            "action": self.action[index],
            "done": self.done[index]
            }
    
    def __len__(self):
        return len(self.action)
