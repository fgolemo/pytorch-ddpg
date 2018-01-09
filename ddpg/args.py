import argparse

import yaml
from ddpg.util import *


class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='PyTorch DDPG')

        self.parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
        self.parser.add_argument('--env', default='Reacher2-v1', type=str, help='open-ai gym environment')
        self.parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
        self.parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
        self.parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
        self.parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
        self.parser.add_argument('--warmup', default=100, type=int,
                                 help='time without training but only filling the replay memory')
        self.parser.add_argument('--discount', default=0.99, type=float, help='')
        self.parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
        self.parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
        self.parser.add_argument('--window_length', default=1, type=int, help='')
        self.parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
        self.parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
        self.parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
        self.parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
        self.parser.add_argument('--validate_episodes', default=20, type=int,
                                 help='how many episode to perform during validate experiment')
        self.parser.add_argument('--max_episode_length', default=500, type=int, help='')
        self.parser.add_argument('--validate_steps', default=2000, type=int,
                                 help='how many steps to perform a validate experiment')
        self.parser.add_argument('--output', default='rl-logs', type=str, help='')
        self.parser.add_argument('--debug', dest='debug', action='store_true')
        self.parser.add_argument('--vis', dest='vis', action='store_true')
        self.parser.add_argument('--init_w', default=0.003, type=float, help='')
        self.parser.add_argument('--train_iter', default=200000, type=int, help='train iters each timestep')
        self.parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
        self.parser.add_argument('--seed', default=-1, type=int, help='')
        self.parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
        self.parser.add_argument('--best', dest='best', action='store_true',
                                 help='If set load best policy instead of last policy')

    def get_args(self, env=None):

        args = self.parser.parse_args()
        if env is not None:
            args.env = env

        args.output = get_output_folder(args.output, args.env, args.resume)
        if args.resume == 'default':
            args.resume = 'output/{}-run0'.format(args.env)

        if args.mode == "train":
            with open(args.output + "/params.yml", "w") as cfg_file:
                yaml.dump(vars(args), cfg_file, default_flow_style=False)

        return args

