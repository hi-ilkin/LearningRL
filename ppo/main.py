import os
import sys

import gym
import numpy as np
from models import Agent
from utils import plot_learning_curve

# define hyper-params
learn_every = 20
batch_size = 5
n_epochs = 4
alpha = 0.0003
n_games = 300
show_every = 50

figure_file = 'plots/cartpole.png'
os.makedirs('plots', exist_ok=True)


def play(agent, env, games=3):
    agent.load_models()

    for i in range(games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            action, prob, value = agent.choose_action(observation)
            new_observation, reward, done, info = env.step(action)
            score += reward
            observation = new_observation
            env.render()

        print(f'Reward of episode {i} is {score}')


def train(agent, env):
    best_score = -np.inf
    score_history = []

    learn_iters = 0
    n_steps = 0
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            action, prob, value = agent.choose_action(observation)
            new_observation, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, value, reward, done)
            if n_steps % learn_every == 0:
                agent.learn()
                learn_iters += 1
            else:
                if i % show_every == 0:
                    env.render()
            observation = new_observation

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)


def main(args):
    # create env
    env = gym.make('CartPole-v0')

    # create agent
    agent = Agent(n_actions=env.action_space.n,
                  input_dims=env.observation_space.shape,
                  alpha=alpha,
                  n_epochs=n_epochs,
                  batch_size=batch_size)

    if args[1] == 'train':
        train(agent, env)
    elif args[1] == 'play':
        play(agent, env)
    else:
        print('Please choose train or playing mode.')


if __name__ == '__main__':
    main(sys.argv)
