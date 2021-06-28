import torch.utils.data as torch_data


class Dataset(torch_data.Dataset):
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
