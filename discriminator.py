import torch
import torch.nn as nn
import torch.optim as optim


class Discriminator(nn.Module):
    """the discriminator class
    """
    def __init__(self, state_shape, hidden_shape=100, learning_rate=0.0002):

        super(Discriminator, self).__init__()
        input_shape = state_shape + 1
        self.main = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=hidden_shape, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=hidden_shape, out_features=hidden_shape, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=hidden_shape, out_features=1, bias=True), # final
        )
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, input):
        return self.main(input.float())

    def train(self, expert_trajectories, policy_trajectories):
        """train to distinguish expert from generated data
         Args:
            expert_trajectories (List)
            policy_trajectories ([type])
        Return:
            error, mean of the predicted score for expert samples and same for the generator samples
        """
        criterion = nn.BCEWithLogitsLoss()
        expert_output = torch.cat([self.forward(torch.cat((state, action))) 
                            for state, action in zip(expert_trajectories['state'], torch.unsqueeze(expert_trajectories['action'], 1))])
        policy_output = torch.cat([self.forward(torch.cat((state, action))) 
                            for state, action in zip(policy_trajectories['state'], torch.unsqueeze(policy_trajectories['action'], 1))])

        expert_labels = torch.zeros_like(expert_output, dtype=torch.float)
        policy_labels = torch.ones_like(policy_output, dtype=torch.float)

        errD_expert = criterion(expert_output, expert_labels)
        errD_policy = criterion(policy_output, policy_labels)

        errD_expert.backward()
        errD_policy.backward()

        errD = errD_expert + errD_policy
        self.optimizer.step()

        with torch.no_grad():
            expert_output = torch.sigmoid(expert_output)
            policy_output = torch.sigmoid(policy_output)

        return errD.item(), torch.mean(expert_output).item(), torch.mean(policy_output).item(), expert_output, policy_output
    