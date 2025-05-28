import time
import numpy as np
import math
import random
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable
import wandb
from Algorithm import GFN as Policy
from itertools import count
from pathlib import Path
import os
import ast

def _t2n(x):
    return x.detach().cpu().numpy()

class GFNRunner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.env = config['env']
        self.eval_env = config['eval_env']
        self.device = config['device']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.num_env_steps = self.all_args.earthquake_num_env_steps
        print("num_env_steps {}".format(self.num_env_steps))
        self.update_interval = 400
        self.episode_length = self.all_args.episode_length
        self.episodes = 1059
        print("epiosdes {}".format(self.episodes))
        self.hidden_dim = self.all_args.hidden_dim
        self.use_wandb = self.all_args.use_wandb
        self.tau_worker = self.all_args.tau_worker #1.0
        self.tau_manager = self.all_args.tau_manager #1.0
        self.gamma_worker = self.all_args.gamma_worker #0.95
        self.gamma_manager = self.all_args.gamma_manager #0.99
        self.alpha = self.all_args.alpha #0.8
        self.entropy_coef = self.all_args.entropy_coef #0.01
        self.value_worker_loss_coef = self.all_args.value_worker_loss_coef #1
        self.value_manager_loss_coef = self.all_args.value_manager_loss_coef #1
        self.max_grad_norm = self.all_args.max_grad_norm #40
        # log
        self.num_act_nodes = []
        self.msg = []
        self.acc_rewards = []
        self.train_info = {}
        self.train_info['epi_loss'] = 0
        self.train_info['inf'] = 0
        self.train_info['msg'] = 0
        self.train_info['acc_rewards'] = 0
        self.epoch_loss = []
        self.epoch_influence = []
        self.epoch_msg = []
        self.epoch_acc_rewards = 0
        self.eval_info = {}
        self.eval_info['num_act_nodes'] = 0
        self.eval_info['msg_effi'] = 0
        self.max_epoch_influence = 0
        self.min_epoch_msgeffi = 5

        # interval
        self.save_interval = self.all_args.save_interval
        print("save_interval {}".format(self.save_interval))
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        print("eval_interval {}".format(self.eval_interval))
        self.log_interval = self.all_args.log_interval
        print("log_interval {}".format(self.log_interval))

        # dir
        self.model_dir = None
        curr_dir = os.path.dirname(__file__)
        self.save_dir = str(wandb.run.dir)
        self.run_dir = str(wandb.run.dir)

        # Feudal network
        self.policy_net = Policy(self.env.num_feats)
        print("{} seeds".format(self.env.num_seeds))
        print("msg_limit {}".format(self.env.l))

        if self.model_dir != None:
            policy_net_state_dict = torch.load(self.model_dir)
            self.policy_net.load_state_dict(policy_net_state_dict)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.all_args.lr)

    
    def log_train(self, total_num_steps):
        self.train_info['epi_loss'] = np.sum(self.epoch_loss)
        self.train_info['inf'] = np.sum(self.epoch_influence)
        self.train_info['msg'] = np.sum(self.epoch_msg)
        self.train_info['acc_rewards'] = self.epoch_acc_rewards
        for k, v in self.train_info.items():
            wandb.log({k: v}, step=total_num_steps)
        self.epoch_loss = []
        self.epoch_influence = []
        self.epoch_msg = []
        self.epoch_acc_rewards = 0


    def log_eval(self, total_num_steps):
        self.eval_info['num_act_nodes'] = np.sum(self.epoch_influence)
        self.eval_info['msg_effi'] = float(np.sum(self.epoch_msg)) / self.eval_info['num_act_nodes']
        for k, v in self.eval_info.items():
            wandb.log({k: v}, step=total_num_steps)
        self.epoch_influence = []
        self.epoch_msg = []


    def run(self):
        episodes = int(self.num_env_steps) // self.episode_length
        episode_start = 0
        training = True

        for episode in range(episode_start, episodes):
            sg_id = episode
            epoch = '70%'
            obs = self.env.reset(epoch, sg_id)
            terminated = False
            episode_step = 0

            while(episode_step < self.episode_length):
                values_worker, values_manager = [], []
                log_probs = []
                rewards = []
                entropies = []  # regularisation
                manager_partial_loss = []

                g_node = []

                for _ in range(1000):
                    value_worker, value_manager, a_values, tran_grad = self.policy_net(obs, training)
                    if a_values == None:
                        edge_index = None
                    else:
                        action_probs = F.softmax(a_values, dim=1)
                        m = Categorical(probs=action_probs)
                        edge_index = m.sample()
                        log_prob = m.log_prob(edge_index)
                        log_probs.append(log_prob)
                        entropy = -(log_prob * action_probs).sum(1, keepdim=True)
                        entropies.append(entropy)
                        manager_partial_loss.append(tran_grad)

                    obs, reward, terminated = self.env.step(g_node, edge_index)
                    self.epoch_acc_rewards += reward
                    if a_values != None:
                        values_manager.append(value_manager)
                        values_worker.append(value_worker)
                        rewards.append(reward)

                    if terminated:
                        break

                episode_step += 1000
                active_nodes = list(set(self.env.active_nodes))
                self.epoch_influence.append(len(active_nodes))
                self.epoch_msg.append(self.env.ep_msg_num)

                if terminated:
                    self.num_act_nodes.append(len(active_nodes))
                    self.msg.append(self.env.ep_msg_num)
                    self.acc_rewards.append(self.epoch_acc_rewards)
                    R_worker = torch.zeros(1, 1).to(self.device)
                    R_manager = torch.zeros(1, 1).to(self.device)
                # The policy and the value function are updated after every t_max actions or when a terminal state is reaches
                else:
                    value_worker, value_manager, _, _ = self.policy_net(obs, training) # Bootstrap from last state
                    R_worker = value_worker
                    R_manager = value_manager

                values_worker.append(Variable(R_worker))
                values_manager.append(Variable(R_manager))
                policy_loss = 0
                manager_loss = 0
                value_manager_loss = 0
                value_worker_loss = 0
                gae_worker = torch.zeros(1, 1).to(self.device)
                for i in reversed(range(len(rewards))):
                    R_worker = self.gamma_worker * R_worker + rewards[i]
                    R_manager = self.gamma_manager * R_manager + rewards[i]
                    advantage_worker = R_worker - values_worker[i]
                    advantage_manager = R_manager - values_manager[i]
                    value_worker_loss = value_worker_loss + 0.5 * advantage_worker.pow(2)
                    value_manager_loss = value_manager_loss + 0.5 * advantage_manager.pow(2)

                    # Generalized Advantage Estimation
                    delta_t_worker = rewards[i] \
                        + self.gamma_worker * values_worker[i + 1].data \
                        - values_worker[i].data

                    gae_worker = gae_worker * self.gamma_worker * self.tau_worker + delta_t_worker

                    policy_loss = policy_loss \
                        - log_probs[i] * gae_worker - self.entropy_coef * entropies[i]

                    if i < len(rewards):
                        # TODO try padding the manager_partial_loss with end values (or zeros)
                        manager_loss = manager_loss \
                            - advantage_manager * manager_partial_loss[i]

                total_loss = policy_loss \
                    + manager_loss \
                    + self.value_manager_loss_coef * value_manager_loss \
                    + self.value_worker_loss_coef * value_worker_loss

                if len(rewards) != 0:
                    self.epoch_loss.append(total_loss.item())

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                if episode % self.log_interval == 0:
                    print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, acc_rewards {}, inf {}, msg {}.\n"
                            .format(self.all_args.scenario,
                                    self.algorithm_name,
                                    self.experiment_name,
                                    episode,
                                    episodes,
                                    self.epoch_acc_rewards,
                                    np.sum(self.epoch_influence),
                                    np.sum(self.epoch_msg)))

                    self.log_train(episode+1)

                if (episode+1) % self.save_interval == 0:
                    torch.save(self.policy_net.state_dict(), str(self.save_dir) + "/GFN_" + str(episode+1) + ".pt")

                if terminated:
                    break

            if episode == episodes - 1:
                np.savetxt(str(self.save_dir) + '/inf.txt', np.array(self.num_act_nodes), fmt='%d')
                np.savetxt(str(self.save_dir) + '/msg.txt', np.array(self.msg), fmt='%d')
                np.savetxt(str(self.save_dir) + '/acc_rewards.txt', np.array(self.acc_rewards), fmt='%f')
