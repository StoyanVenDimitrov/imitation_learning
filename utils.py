"""Functions for preparing and visualising GAIL training
"""
import datetime

import matplotlib.pyplot as plt
import torch
import torch.utils.data as torch_data

from dataset import ExpertDataset, PolicyDataset


def _generate_expert_demonstrations(model, env, trajectories):
    """generate demonstrations with an expert model for some timesteps
    Args:
        model (Generator): generator
        trajectories (int): how many trajectories to generate
    Returns:
        [dict]: trajectories
    """
    obs = env.reset()
    flat_trajectories = {key: [] for key in ["state", "action", "done"]}
    for i in range(trajectories):
        done = False
        while not done:
            flat_trajectories["state"].append(obs)
            action = model.predict(torch.as_tensor(obs, dtype=torch.float32))
            flat_trajectories["action"].append(action)
            obs, reward, done, info = env.step(action)
            flat_trajectories["done"].append(done)
            env.render()
            if done:
                obs = env.reset()
    return flat_trajectories

def _generate_policy_demonstrations(generator, env, gen_timesteps):
    """generate demonstrations with the current generator for some timesteps
    Args:
        model (Generator): generator
        timesteps (int): how many steps to generate
    Returns:
        [dict]: trajectories
    """
    o, ep_len = env.reset(), 0
    flat_trajectories = {key: [] for key in ["state", "next_state", "action", "reward", "value", "logprop", "done"]}
    pairs = []
    for t in range(gen_timesteps):
        terminal = False
        flat_trajectories["state"].append(o)
        a, v, logp = generator.policy.step(torch.as_tensor(o, dtype=torch.float32))
        next_o, r, d, _ = env.step(a)

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
        if ep_len == generator.max_ep_len-1:
            terminal = True
        if d or terminal:
            o, ep_len = env.reset(), 0
    
    return flat_trajectories, pairs


def create_dataloader(trajectories, batch_size, expert):
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


def _reward_statistics(data, expert_mean):
    """get some reward statistics from current iteration

    Args:
        data ([type]): the current trajectories
    """
    std_dev = torch.std(torch.cat([i['reward'] for i in data]))
    try:
        less_than_expert_mean = torch.cat([i['reward'] for i in data if i['reward'] > -expert_mean]).shape[0]
    except NotImplementedError:
        less_than_expert_mean = 0
    return std_dev, less_than_expert_mean


def _draw_gen_result(iters, avg_len, exp_avg, probe_avg_ep_len=None, avg_rew_ep_len=None):
    exp_avg_len = [exp_avg]*len(avg_len)
    if probe_avg_ep_len:
        plt.plot(iters, probe_avg_ep_len, '-y', label='Avg episode length (original reward)')
    if avg_rew_ep_len:
        plt.plot(iters, probe_avg_ep_len, '-c', label='Avg episode length (avgeraged reward)')
    plt.plot(iters, avg_len, '-b', label='Average episode length')
    # plt.plot(iters, avg_ret, '-r', label='Average episode return')
    plt.plot(iters, exp_avg_len, '-y', label='Avg episode length (expert)')

    plt.xlabel("n iterations")
    plt.grid(color='g', linestyle='-', linewidth=0.1)
    plt.legend(loc='upper left')
    plt.title("Generator")

    # save image
    date = datetime.datetime.utcnow().strftime("%H:%M:%S_%b_%d_")
    plt.savefig(f'plots/generator_{date}.png')  # should before show method

def _draw_disc_result(batches, policy_mean, expert_mean, disc_losses):
    plt.plot(batches, policy_mean, '-b', label='mean policy score')
    plt.plot(batches, expert_mean, '-r', label='mean expert score')
    plt.plot(batches, disc_losses, '-y', label='discriminator loss')

    plt.xlabel("n batches")
    plt.grid(color='g', linestyle='-', linewidth=0.1)
    plt.legend(loc='upper left')
    plt.title("Discriminator")

    # save image
    date = datetime.datetime.utcnow().strftime("%H:%M:%S_%b_%d_")
    plt.savefig(f'plots/discriminator_{date}.png')  # should before show method

def _draw_score_statitics(iterations, rew_dev, expert_scores):
    """draw statistics for the rewards assignet to the generated trajectories

    Args:
        iterations ([list]): train iterations 
        rew_dev (list): std deviation of the predicted scores
        expert_scores (list): number of scores less than the avg expert score
    """
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(iterations, rew_dev)

    ax1.set(xlabel='iterations', ylabel='rewards std dev')
    ax2.plot(iterations, expert_scores)

    ax2.set(xlabel='iterations', ylabel='num of expert scores')
    fig.suptitle('Reward statistics')

    # save image
    date = datetime.datetime.utcnow().strftime("%H:%M:%S_%b_%d_")
    fig.savefig(f'plots/rewards_{date}.png')  # should before show method

def _expert_avg_len(expert_trajectories):
    """compute the average episode length for the expert demonstrations
    Args:
        expert_trajectories ([type]): the generate expert demonstrations
    """
    done_at = [i for i,d in  enumerate(expert_trajectories['done']) if d == True or i == 0]
    lengths = [t - s for s, t in zip(done_at, done_at[1:])]
    return sum(lengths)/len(lengths)
