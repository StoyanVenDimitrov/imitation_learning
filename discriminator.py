from numpy import exp
import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.modules.activation import ReLU


class Discriminator(nn.Module):
    """the discriminator class
    """
    def __init__(self, state_shape, action_shape, hidden_shape=128):

        super(Discriminator, self).__init__()
        input_shape = state_shape + 1
        self.main = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=hidden_shape, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hidden_shape, out_features=hidden_shape, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hidden_shape, out_features=1, bias=True), # final
            # nn.Sigmoid() # use BCEWithLogitsLoss instead
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.0002)

    def forward(self, input):
        return self.main(input.float())

    def train(self, expert_trajectories, policy_trajectories):
        """train to distinguish expert from generated data
         Args:
            expert_trajectories ([type])
            policy_trajectories ([type])
        """
        criterion = nn.BCEWithLogitsLoss()

        # ! check network input 
        # cat_inputs = th.cat((obs, acts), dim=1)
        
        expert_output = torch.cat([self.forward(torch.cat((state, action))) 
                            for state, action in zip(expert_trajectories['state'], torch.unsqueeze(expert_trajectories['action'], 1))])
        policy_output = torch.cat([self.forward(torch.cat((state, action))) 
                            for state, action in zip(policy_trajectories['state'], torch.unsqueeze(policy_trajectories['action'], 1))])

        expert_labels = torch.zeros_like(expert_output, dtype=torch.float)
        policy_labels = torch.ones_like(policy_output, dtype=torch.float)


        errD = criterion(expert_output, expert_labels) + criterion(policy_output, policy_labels)
        errD.backward()
        self.optimizer.step()

        return errD
    