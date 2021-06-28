# actor-critic implementation taken from pytorch/examples

import gym
from example import select_action
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.utils.data as torch_data

from dataset import Dataset

class Generator:
    """generator with actor-critic policy
    """
    def __init__(self, env, state_shape, num_actions, discriminator) -> None:
        self.policy_net = PolicyNet(state_shape, num_actions)
        self.env = env
        self.discriminator = discriminator
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-2)
        self.gamma = 0.99

    def generate_rollouts(self, num_of_steps):
        """generate episodes with current policy on some environment 
        """
        trajectories = {key: [] for key in ["state", "next_state", "action", "done"]}
        state = self.env.reset()

        for _ in range(num_of_steps):
            action, _, _ = self.policy_net.select_action(state)
            next_state, _, done, _ = self.env.step(action)
            trajectories['action'].append(action)
            trajectories['state'].append(state)
            trajectories['next_state'].append(next_state)
            trajectories['done'].append(done)
            state = next_state
            if done:
                state = self.env.reset()

        dataset = Dataset(trajectories)
        data_loader = torch_data.DataLoader(
                dataset,
                batch_size=32,
                shuffle=True,
                drop_last=True,
            )
        # print(next(iter(expert_data_loader)))
        return data_loader

    def train(self, num_of_episodes, num_of_steps):
        """train to maximize discriminator loss
        """
        running_reward = 10

        # train the policy for some time:
        for i in range(num_of_episodes):
            ep_reward = 0
            log_probs_buffer = []
            rewards = []
            values_buffer = []

            state = self.env.reset()

            for _ in range(num_of_steps):
                action, log_prob, value = self.policy_net.select_action(state)
                next_state, _, done, _  = self.env.step(action)
                # next_state, _, done, _  = self.env(action)
                # actions.append(action)
                # states.append(state)
                disc_input = np.concatenate([state, [action]])
                reward = self.discriminator.forward(torch.from_numpy(disc_input))
                rewards.append(reward)
                ep_reward += reward
                log_probs_buffer.append(log_prob)
                values_buffer.append(value)
                state = next_state
                if done:
                    break   
            
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
                
            self._train_opt(log_probs_buffer, values_buffer, rewards)

            # log results
            if i % 10 == 0:
                print(f'Episode {i}\tLast reward: {ep_reward}\tAverage reward: {running_reward}')

            # # check if we have "solved" the cart pole problem
            # if running_reward > self.env.spec.reward_threshold:
            #     print("Solved! Running reward is now {} and "
            #         "the last episode runs to {} time steps!".format(running_reward, t))
            #     break

    def _train_opt(self, log_probs, values, rewards):
        """after the episode, do one step of actor-critic update
        Args:
            log_probs (list): the log-probability of the choosen action
            values (list): the state value
            rewards (list): the reward for the transition 
        """
        R = 0
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)

        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for log_prob, value, R in zip(log_probs, values, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss 
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        self.optimizer.step()
        


    def train_ppo(self):
        """train using PPO (implemented here) 
        (Basically, the sb3 PPO but with loss, containing the disc. output instead advantage function.
        Replacing A with Q(s,a) changes the expression only by a constant (look TRPO, sec.5))
        """


class PolicyNet(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, state_shape, num_actions):
        super(PolicyNet, self).__init__()
        self.affine1 = nn.Linear(state_shape, 128)

        # actor's layer
        self.action_head = nn.Linear(128, num_actions)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(state))

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_value = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return action_prob, state_value


    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probs, state_value = self.forward(state)

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)

        # and sample an action using the distribution
        action = m.sample()

        # the action to take, its log prob and the state value
        return action.item(), m.log_prob(action), state_value


def main():
    env = gym.make('CartPole-v0')
    env.seed(543)
    generator = Generator(env, 4, 2, None)
    generator.train(420, 10000)
    actions, states, dones = generator.generate_rollouts(1000)
    print(dones)

if __name__ == '__main__':
    main()