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
from generator import Generator
from discriminator import  Discriminator

from dataset import Dataset

EXPERT_TIMESTEPS = 10000
GENERATOR_TIMESTEPS = 1000
EXPERT_TRAIN_EPOCHS = 5

BATCH_SIZE = 32
EPOCHS = 7

class GAIL:
    """Class for training the GAIL algorithm 
    """

    def __init__(self, env_name) -> None:
        # self.env = gym.make(env_name)
        self.env = gym.make(env_name)
        self.env.seed(543)
        self.discriminator = Discriminator(state_shape=self.env.observation_space.shape[0], action_shape=self.env.action_space.n)
        self.generator = Generator(self.env, None)


    def get_demonstrations(self,  expert=False):
        env_name = self.env.spec.id
        if expert:
            try:
                with open(f'expert_{env_name}_demos.json', 'r') as fp:
                    flat_trajectories = json_tricks.load(fp)
            except FileNotFoundError:
                model = Generator(self.env, None)
                model.ppo(epochs=EXPERT_TRAIN_EPOCHS)
                flat_trajectories = self.generate_demonstrations(model, EXPERT_TIMESTEPS)
                with open(f'expert_{env_name}_demos.json', 'w') as fp:
                    json_tricks.dump(flat_trajectories, fp)

        if not expert:
            flat_trajectories = self.generate_demonstrations(self.generator,GENERATOR_TIMESTEPS)

        assert np.array_equal(flat_trajectories["state"][1], flat_trajectories["next_state"][0])
        assert np.array_equal(flat_trajectories["state"][2], flat_trajectories["next_state"][1])
        # assert np.array_equal(flat_trajectories["state"][-1], flat_trajectories["next_state"][-2])
        expert_dataset = Dataset(flat_trajectories)
        expert_data_loader = torch_data.DataLoader(
                expert_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True,
            )
        return expert_data_loader
    
    def generate_demonstrations(self, model, timesteps):
            """generate demonstrations with a model for some timesteps
            Args:
                model (Generator): generator
                timesteps (int): how many steps to generate
            Returns:
                [dict]: trajectories
            """
            obs = self.env.reset()
            flat_trajectories = {key: [] for key in ["state", "next_state", "action", "done"]}
            for i in range(timesteps):
                flat_trajectories
                flat_trajectories["state"].append(obs)
                action = model.predict(torch.as_tensor(obs, dtype=torch.float32))
                flat_trajectories["action"].append(action)
                obs, reward, done, info = self.env.step(action)
                flat_trajectories["next_state"].append(obs)
                flat_trajectories["done"].append(done)

                self.env.render()
                if done:
                    obs = self.env.reset()
            return flat_trajectories

        
    def train(self):
        """train alternating the discriminator and the generator

        Args:
            exp_demos ([type]): expert trajectories 
        """

        for epoch in range(EPOCHS):
            # for i, exp_data in enumerate(exp_dataloader, 0):
            # train all until the expert demos are over:
            expert_has_next = True
            exp_dataloader = iter(self.get_demonstrations(expert=True))
            while expert_has_next:
                # generate policy demos and sample batches
                for i, fake_data in enumerate(self.get_demonstrations(expert=False),0):
                    try:
                        # sample expert demos batch
                        exp_data = next(exp_dataloader)
                        # train the discriminator until the policy demos batches are over:
                        disc_loss, expert_mean, policy_mean = self.discriminator.train(exp_data, fake_data)
                        if i % 10 == 0:
                            print(f'Batch {i}\t Discriminator: loss: {disc_loss}\t expert mean {expert_mean} \t generator mean {policy_mean}')
                    except StopIteration:
                        print(f'------------ Epoch {epoch + 1} finished! ------------')
                        expert_has_next = False
                        break
                # train the generator 
            self.generator.ppo(epochs=5)


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
