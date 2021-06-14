import time
import gym
import numpy as np
import torch as torch
import torch.utils.data as torch_data
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


class GAIL:
    """Class for training the GAIL algorithm 
    """

    def __init__(self) -> None:
        # self.env = gym.make(env_name)

        self.generator = Generator()
        self.discriminator = Discriminator()

    def train_expert(self, env_name):
        """train a model to generate expert demons with it
        Args:
            env_name ([type]): [description]
        """
        env = gym.make(env_name)
        model = PPO(
            "MlpPolicy", 
            env, 
            n_epochs=4,
            n_steps=256,
            verbose=1)
        model.learn(total_timesteps=100000)
        model.save(env_name)
        return model

    def get_demonstrations(self, env_name):
        """With the PPO expert policy, create expert demonstrations
        """
        env = gym.make(env_name)
        try:
            env = gym.make(env_name)
            model = PPO.load(env_name, env=env)
        except FileNotFoundError:
            model = self.train_expert(env_name)
        obs = env.reset()
        flat_trajectories = {key: [] for key in ["state", "next_state", "action", "done"]}
        for i in range(100):
            flat_trajectories
            flat_trajectories["state"].append(obs)
            action, _states = model.predict(obs, deterministic=True)
            flat_trajectories["action"].append(action)
            obs, reward, done, info = env.step(action)
            flat_trajectories["next_state"].append(obs)
            flat_trajectories["done"].append(done)

            env.render()
            if done:
                obs = env.reset()
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
        print(next(iter(expert_data_loader)))
        return flat_trajectories
        
    def train(self, exp_demos):
        """train alternating the discriminator and the generator

        Args:
            exp_demos ([type]): expert trajectories 
        """
    
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

class Generator:
    """the generator class
    """
    def __init__(self) -> None:
        pass

    def train_ppo(self):
        """train using PPO (implemented here) 
        (Basically, the sb3 PPO but with loss, containing the disc. output instead advantage function.
        Replacing A with Q(s,a) changes the expression only by a constant (look TRPO, sec.5))
        """
        # 
    
    def train(self):
        """train to maximize discriminator loss
        """

    def predict(self):
        """predict from the policy 
        """

class Discriminator:
    """the discriminator class
    """
    def __init__(self) -> None:
        pass

    def train(self, exp_demos):
        """train to distinguish expert from generated data
         Args:
            exp_demos ([type]): expert trajectories 
        """
    
    def predict(self):
        """predict probability of being expert (HIGH)
        """