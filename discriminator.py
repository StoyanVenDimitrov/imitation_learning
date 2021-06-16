import torch.nn as nn
from torch.nn.modules.activation import ReLU


class Discriminator:
    """the discriminator class
    """
    def __init__(self, input_shape=6, hidden_shape=32):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=hidden_shape, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hidden_shape, out_features=hidden_shape, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hidden_shape, out_features=1, bias=True), # final
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

    def train(self, exp_demos):
        """train to distinguish expert from generated data
         Args:
            exp_demos ([type]): expert trajectories 
        """
    