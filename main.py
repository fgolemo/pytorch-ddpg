#!/usr/bin/env python3 

import argparse
from copy import deepcopy

import gym
import numpy as np

from ddpg.ddpg import DDPG
from ddpg.evaluator import Evaluator
from ddpg.main import train, test
from ddpg.normalized_env import NormalizedEnv
from ddpg.util import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str,
                        help='support option: train/test')
    parser.add_argument('--env', default='Pendulum-v0', type=str,
                        help='open-ai gym environment')
    parser.add_argument('--hidden1', default=400, type=int,
                        help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int,
                        help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float,
                        help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=100, type=int,
                        help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float,
                        help='')
    parser.add_argument('--bsize', default=64, type=int,
                        help='minibatch size')
    parser.add_argument('--rmsize', default=6000000, type=int,
                        help='memory size')
    parser.add_argument('--window_length', default=1, type=int,
                        help='')
    parser.add_argument('--tau', default=0.001, type=float,
                        help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float,
                        help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float,
                        help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float,
                        help='noise mu')
    parser.add_argument('--validate_episodes', default=20, type=int,
                        help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int,
                        help='')
    parser.add_argument('--validate_steps', default=2000, type=int,
                        help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str,
                        help='')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='')
    parser.add_argument('--init_w', default=0.003, type=float,
                        help='')
    parser.add_argument('--train_iter', default=200000, type=int,
                        help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int,
                        help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int,
                        help='')
    parser.add_argument('--resume', default='default', type=str,
                        help='Resuming model path for testing')
    parser.add_argument('--best', dest='best', action='store_true',
                        help='If set load best policy instead of last policy')

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env, args.resume)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)

    env = NormalizedEnv(gym.make(args.env))

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    agent = DDPG(nb_states, nb_actions, args)
    evaluate = Evaluator(args.validate_episodes,
                         args.validate_steps, args.output,
                         max_episode_length=args.max_episode_length)

    if args.mode == 'train':
        train(args, args.train_iter, agent, env, evaluate,
              args.validate_steps, args.output,
              max_episode_length=args.max_episode_length, debug=args.debug)

    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.resume,
             visualize=True, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
