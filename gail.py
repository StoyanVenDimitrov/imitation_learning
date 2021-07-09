from re import A
import time
import gym
import json
import numpy as np
import torch as torch
import torch.utils.data as torch_data
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.evaluation import evaluate_policy
import json_tricks
from torch.utils.data import dataset
from generator import Generator
from discriminator import  Discriminator

from dataset import ExpertDataset, PolicyDataset

EXPERT_TRAJECTORIES = 10
GENERATOR_TIMESTEPS = 1000
EXPERT_TRAIN_EPOCHS = 50
# GENERATOR_TRAIN_EPOCHS = 3 #! must be a single one
BATCH_SIZE = 32
ITERATIONS = 300
MAX_EP_LEN = 1000

class GAIL:
    """Class for training the GAIL algorithm 
    """

    def __init__(self, env_name) -> None:
        # self.env = gym.make(env_name)
        self.env = gym.make(env_name)
        self.env.seed(543)
        self.discriminator = Discriminator(state_shape=self.env.observation_space.shape[0])
        self.generator = Generator(self.env, self.discriminator, max_ep_len=MAX_EP_LEN, steps_per_epoch=GENERATOR_TIMESTEPS)

    def get_demonstrations(self,  expert=False):
        """get demonstrations from an expert/policy model

        Args:
            expert (bool, optional): [are these demonstrations from an expert]. Defaults to False.

        Returns:
            [torch.utils.data.DataLoader]: torch DataLoader object with the created data
        """
        env_name = self.env.spec.id
        if expert:
            try:
                with open(f'expert_{env_name}_demos.json', 'r') as fp:
                    flat_trajectories = json_tricks.load(fp)
            except FileNotFoundError:
                model = Generator(self.env, None)
                model.ppo(epochs=EXPERT_TRAIN_EPOCHS)
                flat_trajectories = self._generate_expert_demonstrations(model, EXPERT_TRAJECTORIES)
                with open(f'expert_{env_name}_demos.json', 'w') as fp:
                    json_tricks.dump(flat_trajectories, fp)
            return flat_trajectories
        if not expert:
            return self._generate_policy_demonstrations()
    
    def create_dataloader(self, trajectories, batch_size, expert):
        """create torch DataLoader with the trajectories

        Args:
            trajectories ([type]): [description]
        Returns:
            [torch.utils.data.DataLoader]: torch DataLoader object with the created data
        """
        if expert:
            dataset = ExpertDataset(trajectories)
        else:
            dataset = PolicyDataset(trajectories)
        dataloader = torch_data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
            )
        return dataloader
       
    def train(self):
        """train alternating the discriminator and the generator

        Args:
            exp_demos ([type]): expert trajectories 
        """
        expert_demos = self.get_demonstrations(expert=True)
        expert_dataloader = self.create_dataloader(expert_demos, BATCH_SIZE, expert=True)
        for iteration in range(ITERATIONS):
            gen_trajectories_flat, gen_pairs = self.get_demonstrations(expert=False)
            gen_dataloader = self.create_dataloader(gen_trajectories_flat, BATCH_SIZE, expert=False) 
            # train the discriminator with batches:
            for i, fake_data in enumerate(gen_dataloader,0):
                exp_data = next(iter(expert_dataloader))
                disc_loss, expert_mean, policy_mean = self.discriminator.train(exp_data, fake_data) 
                if i % 10 == 0:
                    print(f'Batch {i}\t Discriminator: loss: {disc_loss}\t expert mean {expert_mean} \t generator mean {policy_mean}')
            self.generator.ppo(data=gen_pairs)     

            # while policy_has_next:
            #     # generate policy demos and sample batches
            #     for i, fake_data in enumerate(self.get_demonstrations(expert=True),0):
            #         try:
            #             # sample expert demos batch
            #             exp_data = next(iter(self.get_demonstrations(expert=True)))
            #             # train the discriminator until the policy demos batches are over:
            #             disc_loss, expert_mean, policy_mean = self.discriminator.train(exp_data, fake_data)
            #             if i % 10 == 0:
            #                 print(f'Batch {i}\t Discriminator: loss: {disc_loss}\t expert mean {expert_mean} \t generator mean {policy_mean}')
            #         except StopIteration:
            #             policy_has_next = False
            #             break
            #     # train the generator 
            # self.generator.ppo() # (epochs=GENERATOR_TRAIN_EPOCHS)
            # print(f'------------ Iteration {iteration + 1} finished! ------------')

    def _generate_expert_demonstrations(self, model, trajectories):
            """generate demonstrations with an expert model for some timesteps
            Args:
                model (Generator): generator
                trajectories (int): how many trajectories to generate
            Returns:
                [dict]: trajectories
            """
            obs = self.env.reset()
            done = False
            flat_trajectories = {key: [] for key in ["state", "action", "done"]}
            for i in range(trajectories):
                while not done:
                    flat_trajectories["state"].append(obs)
                    action = model.predict(torch.as_tensor(obs, dtype=torch.float32))
                    flat_trajectories["action"].append(action)
                    obs, reward, done, info = self.env.step(action)
                    flat_trajectories["done"].append(done)

                    self.env.render()
                    if done:
                        obs = self.env.reset()
            return flat_trajectories

    def _generate_policy_demonstrations(self):
            """generate demonstrations with an expert model for some timesteps
            Args:
                model (Generator): generator
                timesteps (int): how many steps to generate
            Returns:
                [dict]: trajectories
            """
            o, ep_len = self.env.reset(), 0
            flat_trajectories = {key: [] for key in ["state", "next_state", "action", "reward", "value", "logprop", "done"]}
            pairs = []
            for t in range(GENERATOR_TIMESTEPS):
                terminal = False
                flat_trajectories["state"].append(o)
                a, v, logp = self.generator.policy.step(torch.as_tensor(o, dtype=torch.float32))
                next_o, r, d, _ = self.env.step(a)

                flat_trajectories["action"].append(a)
                flat_trajectories["next_state"].append(next_o)
                flat_trajectories["done"].append(d)
                flat_trajectories["value"].append(v)
                flat_trajectories["logprop"].append(logp)
                flat_trajectories["reward"].append(r)
                pairs.append(
                    {
                        "state": o,
                        "action": a,
                        "next_state": next_o,
                        "done": d,
                        "value": v,
                        "logprop": logp,
                        "reward": r,
                    }
                )
                ep_len += 1
                # Update obs (critical!)
                o = next_o
                if ep_len == self.generator.max_ep_len-1:
                    terminal = True
                if d or terminal:
                    o, ep_len = self.env.reset(), 0
            
            return flat_trajectories, pairs