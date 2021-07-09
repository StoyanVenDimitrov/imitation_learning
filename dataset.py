import torch.utils.data as torch_data


class ExpertDataset(torch_data.Dataset):
    def __init__(self, flat_trajectories):
        self.state = flat_trajectories["state"]
        self.action = flat_trajectories["action"]
        self.done = flat_trajectories["done"]
        
    def __getitem__(self, index):
        return {
            "state": self.state[index], 
            "action": self.action[index],
            "done": self.done[index]
            }
    
    def __len__(self):
        return len(self.action)

class PolicyDataset(torch_data.Dataset):
    def __init__(self, flat_trajectories):
        self.state = flat_trajectories["state"]
        self.next_state = flat_trajectories["next_state"]
        self.action = flat_trajectories["action"]
        self.value = flat_trajectories["value"]
        self.reward = flat_trajectories["reward"]
        self.logprop = flat_trajectories["logprop"]
        self.done = flat_trajectories["done"]
        
    def __getitem__(self, index):
        return {
            "state": self.state[index], 
            "action": self.action[index], 
            "value": self.value[index],
            "reward": self.reward[index],
            "next_state": self.next_state[index],
            "logprop": self.logprop[index],
            "done": self.done[index]
            }
    
    def __len__(self):
        return len(self.action)