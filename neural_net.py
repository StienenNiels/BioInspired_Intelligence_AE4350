import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Neural_network(nn.Module): 
    # inherit the nn.Module class for backpropagation and training functionalities
    
    # Create the layers of the network, and initialize the parameters
    def __init__(self, input_dim: int, output_dim: int):
        """
        Basic Neural network

        args:
        input_dim: int: dimension of the input
        output_dim: int: dimension of the output
        """
        super(Neural_network, self).__init__()
        hidden_dim = 128
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, output_dim)

    # Build the forward pass of the network
    def forward(self, x):
        # Check if x is a tensor, if not convert it to a tensor
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        x = F.tanh(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x