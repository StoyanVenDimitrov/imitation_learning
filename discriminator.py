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
        self.optimizer = optim.Adam(self.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def forward(self, input):
        return self.main(input.float())

    def train(self, expert_trajectories, policy_trajectories):
        """train to distinguish expert from generated data
         Args:
            expert_trajectories ([type])
            policy_trajectories ([type])
        """
        criterion = nn.BCEWithLogitsLoss()

        # ! forward input is single (s,a)
        expert_output = self.forward(expert_trajectories)
        policy_output = self.forward(policy_trajectories)

        expert_labels = torch.zeros_like(expert_output, dtype=torch.float)
        policy_labels = torch.ones_like(policy_output, dtype=torch.float)


        errD = criterion(expert_output, expert_labels) + criterion(policy_output, policy_labels)
        errD.backward()
        self.optimizer.step()

        return errD
    