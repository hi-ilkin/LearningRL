"""
policy is a probability distribution.
p(a|s) - given state s what is the probability of taking action a
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class PPOMermory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        return np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.actions), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super().__init__()
        # handle checkpoint file
        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')

        # create actor model
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        # add optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        # map to GPU if available
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def feed_forward(self, state):
        # Pass current state through the actor model
        dist = self.actor(state)

        # convert probability distribution to categorical values to choose action later
        dist = Categorical(dist)

        return dist

    # Can we use another base class to handle save/load?
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    # we don't need the number of actions because it outputs single value - value of current state
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super().__init__()

        # handle file
        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, 'torch_ppo_critic')

        # create critic model
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        # add optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        # handle using gpu
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def feed_forward(self, state):
        # pass through network to generate value
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.1, batch_size=64, n_horizon=2048, n_epochs=10):
        """
        These numbers are either coming from the original paper or experiments

        :param n_actions: Action space count
        :param input_dims: dimension of the model inputs
        :param gamma: discount factor
        :param alpha: learning rate
        :param gae_lambda: smoothing factor
        :param policy_clip: hyperparameter for policy clipping
        :param batch_size:
        :param n_horizon: horizon - number of steps before doing update
        :param n_epochs:
        """

        # save params
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.gamma = gamma
        self.alpha = alpha
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.n_horizon = n_horizon
        self.n_epochs = n_epochs

        # create actor model
        self.actor = ActorNetwork(n_actions, input_dims, alpha)

        # create critic model
        self.critic = CriticNetwork(input_dims, alpha)

        # create memory
        self.memory = PPOMermory(batch_size)

    # Next three simple models are simple functions to handle interface between agent and memory, actor, critic
    def remember(self, state, action, prob, val, reward, done):
        # handles interface between agent and its memory
        self.memory.store_memory(state, action, prob, val, reward, done)

    def save_models(self):
        print('... Saving models ...', end='')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        print('DONE!')

    def load_models(self):
        print('... Loading models ...', end='')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        print('DONE!')

    def choose_action(self):
        ...
