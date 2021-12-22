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
        """
        Generates random batch from the memory, size of batch_size.

        :return: states, actions, probs, vals, rewards, dones, batches
        """

        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
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

    def forward(self, state):
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

    def forward(self, state):
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
        self.c1 = 0.5  # hyperparameter for Critic loss
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

    def choose_action(self, observation):
        """
        Samples an action based the observation (state)
        :param observation: Current observation
        :return: action, probs, value
        """
        # cast np array to torch.tensor.
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)

        # get action distribution from actor
        dist = self.actor(state)

        # get value from the critic
        value = self.critic(state)

        # sample action from distribution
        action = dist.sample()

        # get probs, action, value
        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value

    def learn(self):
        ...
        for _ in range(self.n_epochs):
            # generate batches
            state_arr, action_arr, prob_arr, value_arr, reward_arr, done_arr, batches = self.memory.generate_batches()

            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            # TODO: GO OVER THIS CALCULATION AGAIN.
            # This is the implementation suggested by the original paper
            # calculate advantage for each t . We are considering k + 1 value that's why len(reward_arr) - 1
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (
                            reward_arr[k] + self.gamma * value_arr[k + 1] * (1 - int(done_arr[k])) - value_arr[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device)
            values = torch.tensor(value_arr).to(self.actor.device)

            # get states, probs and actions for each batch, convert to tensor and map to device
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(prob_arr[batch], dtype=torch.float).to(self.actor.device)
                actions = torch.tensor(action_arr[batch], dtype=torch.float).to(self.actor.device)

                # get new action distribution, critic value and new probabilities
                dist = self.actor(states)
                critic_value = torch.squeeze(self.critic(states))
                new_probs = dist.log_prob(actions)

                # calculate actor loss
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                # TODO:  shouldn't it be weighted_probs instead of prob_ratio?
                weighted_clipped_probs = torch.clip(prob_ratio, 1 - self.policy_clip,
                                                    1 + self.policy_clip) * advantage[batch]

                # We do gradient ascent, that's why negative
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()  # TODO: Why mean?

                returns = advantage[batch] + values[batch]

                # calculate critic loss
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                # total loss
                total_loss = actor_loss + self.c1 * critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
