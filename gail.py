import datetime
import gym
import json_tricks
import matplotlib.pyplot as plt
import torch as torch
import torch.utils.data as torch_data

from dataset import ExpertDataset, PolicyDataset
from discriminator import Discriminator
from generator import Generator

EXPERT_TRAJECTORIES = 7
EXPERT_TRAJECTORIES_LIST = [1, 4, 7, 10]
GENERATOR_TIMESTEPS = 5000
EXPERT_TRAIN_EPOCHS = 50
BATCH_SIZE = 32
ITERATIONS = 100 #300
MAX_EP_LEN = 1000
DISC_ITER = -1
DISC_L_RATE = 0.0002

class GAIL:
    """Class for training the GAIL algorithm 
    """

    def __init__(self, env_name) -> None:
        # self.env = gym.make(env_name)
        self.env = gym.make(env_name)
        self.env.seed(543)
        self.discriminator = Discriminator(state_shape=self.env.observation_space.shape[0], learning_rate=DISC_L_RATE)
        self.generator = Generator(self.env, self.discriminator, max_ep_len=MAX_EP_LEN, steps_per_epoch=GENERATOR_TIMESTEPS)
        self.avg_rew_generator = Generator(self.env, self.discriminator, max_ep_len=MAX_EP_LEN, steps_per_epoch=GENERATOR_TIMESTEPS)
        self.avg_rew_generator.load_state_dict(self.generator.state_dict())
        # make one generator that learns from the original reward
        # self.probe_generator = Generator(self.env, None, max_ep_len=MAX_EP_LEN, steps_per_epoch=GENERATOR_TIMESTEPS)
        # self.probe_generator.load_state_dict(self.generator.state_dict())

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
                with open(f'expert_{env_name}_demos_{EXPERT_TRAJECTORIES}.json', 'r') as fp:
                    flat_trajectories = json_tricks.load(fp)
            except FileNotFoundError:
                model = Generator(self.env, None)
                model.ppo(epochs=EXPERT_TRAIN_EPOCHS)
                for num_trajectories in EXPERT_TRAJECTORIES_LIST:
                    flat_trajectories = self._generate_expert_demonstrations(model, num_trajectories)
                    with open(f'expert_{env_name}_demos_{num_trajectories}.json', 'w') as fp:
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
        if batch_size is None:
            batch_size = dataset.__len__()
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
        # for ploting
        disc_losses, expert_means, policy_means = [], [], []
        avg_ep_len, avg_ep_ret, probe_avg_ep_len = [], [], []
        batches, iterations = [], []
        disc_batch, gen_iter = 1, 1 
        # generate expert demonstrations
        expert_demos = self.get_demonstrations(expert=True)
        # get their avg episode length
        expert_avg_len = self._expert_avg_len(expert_demos)
        # #! Using single batch
        expert_dataloader = self.create_dataloader(expert_demos, None, expert=True)
        batch_size = expert_dataloader.dataset.__len__()
        # expert_dataloader = self.create_dataloader(expert_demos, BATCH_SIZE, expert=True)
        # batch_size = BATCH_SIZE
        for iteration in range(ITERATIONS):
            print(f'###########  Generating trajectories...  #########')
            gen_trajectories_flat, gen_pairs = self.get_demonstrations(expert=False)
            gen_dataloader = self.create_dataloader(gen_trajectories_flat, batch_size, expert=False) 

            # train the discriminator with batches:
            for i, fake_data in enumerate(gen_dataloader,0):
                exp_data = next(iter(expert_dataloader))
                disc_loss, expert_mean, policy_mean, expert_output, policy_output = self.discriminator.train(exp_data, fake_data) 
                if i % 3 == 0:
                    print(f'Batch {i}\t Discriminator: loss: {disc_loss}\t expert mean {expert_mean} \t generator mean {policy_mean}')
                # for the plots:
                expert_means.append(expert_mean)
                policy_means.append(policy_mean)
                disc_losses.append(disc_loss)
                batches.append(disc_batch)
                disc_batch += 1
            # eventually, pretrain the discriinator; default is DISC_ITER = -1
            if iteration > DISC_ITER:
                # train the generator with all generator demonstrations
                self.generator.ppo(data=gen_pairs)  
                # self.probe_generator.ppo()  
                # for the plots:   
                avg_ep_len.append(self.generator.avg_ep_len)
                # probe_avg_ep_len.append(self.probe_generator.avg_ep_len)
                avg_ep_ret.append(self.generator.avg_ep_return)
                iterations.append(gen_iter)
                gen_iter += 1
                print(f'------------ Iteration {iteration + 1} finished! ------------')
        
        self._draw_gen_result(iterations, avg_ep_len, expert_avg_len)#, probe_avg_ep_len)
        plt.show()
        self._draw_disc_result(batches, policy_means, expert_means, disc_losses)
        plt.show()

    def _generate_expert_demonstrations(self, model, trajectories):
            """generate demonstrations with an expert model for some timesteps
            Args:
                model (Generator): generator
                trajectories (int): how many trajectories to generate
            Returns:
                [dict]: trajectories
            """
            obs = self.env.reset()
            flat_trajectories = {key: [] for key in ["state", "action", "done"]}
            for i in range(trajectories):
                done = False
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

    def _draw_gen_result(self, iters, avg_len, exp_avg):#, probe_avg_ep_len):
        exp_avg_len = [exp_avg]*len(avg_len)
        plt.plot(iters, avg_len, '-b', label='Average episode length')
        # plt.plot(iters, probe_avg_ep_len, '-c', label='Average episode length (original reward)')
        # plt.plot(iters, avg_ret, '-r', label='Average episode return')
        plt.plot(iters, exp_avg_len, '-y', label='Average episode length (expert)')

        plt.xlabel("n iteration")
        plt.grid(color='g', linestyle='-', linewidth=0.1)
        plt.legend(loc='upper left')
        plt.title("Generator")

        # save image
        date = datetime.datetime.utcnow().strftime("%H:%M:%S_%b_%d_")
        plt.savefig(f'plots/generator_iter_{EXPERT_TRAJECTORIES}_{date}.png')  # should before show method

    def _draw_disc_result(self, batches, policy_mean, expert_mean, disc_losses):
        plt.plot(batches, policy_mean, '-b', label='mean policy score')
        plt.plot(batches, expert_mean, '-r', label='mean expert score')
        # plt.plot(batches, disc_losses, '-y', label='discriminator loss')

        plt.xlabel("n batches")
        plt.grid(color='g', linestyle='-', linewidth=0.1)
        plt.legend(loc='upper left')
        plt.title("Discriminator")

        # save image
        date = datetime.datetime.utcnow().strftime("%H:%M:%S_%b_%d_")
        plt.savefig(f'plots/discriminator_iter_{EXPERT_TRAJECTORIES}_{date}.png')  # should before show method

    def _expert_avg_len(self, expert_trajectories):
        """compute the average episode length for the expert demonstrations
        Args:
            expert_trajectories ([type]): the generate expert demonstrations
        """
        done_at = [i for i,d in  enumerate(expert_trajectories['done']) if d == True or i == 0]
        lengths = [t - s for s, t in zip(done_at, done_at[1:])]
        return sum(lengths)/len(lengths)
