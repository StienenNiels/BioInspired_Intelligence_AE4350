import torch
from torch import nn
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from torch.distributions import MultivariateNormal
import os
import pandas as pd
import neural_net
import json

class Model_Class:
    def __init__(self, env: gym.Env, seed=123) -> None:
        # Set the seed for reproducibility
        torch.manual_seed(seed)

        # Set the environment and get the action and observation space
        self.env = env
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # Create the actor and critic networks for Soft Actor-Critic (SAC)
        self.actor = neural_net.Neural_network(self.observation_dim, self.action_dim)
        self.critic = neural_net.Neural_network(self.observation_dim, 1)

        # Initialize the hyperparameters
        self._init_hyperparams()

        # Settings for saving and loading the model
        self.trained = False
        self.timestepGlobal = 0
        self.plotRewards = []
        self.plotTimestep = []

        self.record_every = int(512*8)
        self.timestep_of_last_record = -(self.record_every+1)

    def _init_hyperparams(self, config_file='hyperparameters.json'):
         # Read the configuration file
        with open(config_file, 'r') as file:
            params = json.load(file)
        
        # Set hyperparameters from the file (if not provided, set default values)
        self.timesteps_per_batch        = params.get('timesteps_per_batch', 4096)
        self.max_timesteps_per_episode  = params.get('max_timesteps_per_episode', 256)
        self.gamma                      = params.get('gamma', 0.9)
        self.n_updates_per_iteration    = params.get('n_updates_per_iteration', 25)
        self.clip                       = params.get('clip', 0.1)
        self.lr                         = params.get('lr', 0.0002)
        self.max_grad_norm              = params.get('max_grad_norm', 0.8)

    def save(self, path: str):
        print("Saving")
        
        os.makedirs(path, exist_ok=True)

        # Save actor and critic state dict
        torch.save(self.actor.state_dict(), os.path.join(path, 'ppo_actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(path, 'ppo_critic.pth'))

        # Save CSV file
        csv = pd.DataFrame(np.array([self.plotTimestep, self.plotRewards]).T, columns=["timestep", "reward"])
        csv.to_csv(os.path.join(path, f"results.csv"))

    def load(self, path: str):
        print("Loading")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory {path} does not exist")

        # Load actor and critic state dict
        self.actor.load_state_dict(torch.load(os.path.join(path, 'ppo_actor.pth')))
        self.critic.load_state_dict(torch.load(os.path.join(path, 'ppo_critic.pth')))