"""Functions for preparing and visualising GAIL training
"""
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