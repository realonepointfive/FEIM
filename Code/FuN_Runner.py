import time
from networkx import selfloop_edges
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
from Algorithm import FeudalNet as Policy
from ExperienceReplay import Transition, ReplayMemory
from itertools import count
from pathlib import Path
import os

def _t2n(x):
    return x.detach().cpu().numpy()

class FuNRunner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.env = config['env']
        self.eval_env = config['eval_env']
        self.device = config['device']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.num_episodes = self.all_args.num_episodes
        self.eval_episodes = self.all_args.eval_episodes
        self.hidden_dim = self.all_args.hidden_dim
        self.use_wandb = self.all_args.use_wandb
        self.tau_worker = self.all_args.tau
        self.gamma_worker = self.all_args.gamma_worker
        self.gamma_manager = self.all_args.gamma_manager
        self.alpha = self.all_args.alpha
        self.entropy_coef = self.all_args.entropy_coef
        self.value_worker_loss_coef = self.all_args.value_worker_loss_coef
        self.value_manager_loss_coef = self.all_args.value_manager_loss_coef
        self.max_grad_norm = self.all_args.max_grad_norm

        # log
        self.ep_loss = []
        self.ep_acc_rewards = []
        self.ep_num_act_nodes = []
        self.ep_msg_effi = []
        self.train_info = {}
        self.train_info['ep_aver_loss'] = 0
        self.train_info['ep_acc_rewards'] = 0
        self.train_info['ep_num_act_nodes'] = 0
        self.train_info['ep_msg_effi'] = 0
        self.eval_info = {}
        self.eval_info['ep_acc_rewards'] = 0
        self.eval_info['ep_num_act_nodes'] = 0
        self.eval_info['ep_msg_effi'] = 0
        
        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / self.all_args.env_name / self.all_args.scenario / self.all_args.algorithm_name / self.all_args.experiment_name

        self.save_dir = str(wandb.run.dir)
        self.run_dir = str(wandb.run.dir)

        # Feudal network
        self.policy_net = Policy(self.env.num_feats,
                                 self.hidden_dim)
        # self.policy_net.share_memory()
        self.IP = [torch.zeros(self.policy_net.k, requires_grad=False).to(self.device) for _ in range(self.policy_net.h)]
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.all_args.lr)

        if self.model_dir is not None:
            self.restore(self.model_dir)


    def log_train(self, total_num_steps):
        self.train_info['ep_aver_loss'] = np.mean(self.ep_loss)
        self.train_info['ep_num_act_nodes'] = np.mean(self.ep_num_act_nodes)
        self.train_info['ep_msg_effi'] = np.mean(self.ep_msg_effi)
        self.train_info['ep_acc_rewards'] = np.mean(self.ep_acc_rewards)
        self.ep_num_act_nodes = []
        self.ep_loss = []
        self.ep_msg_effi = []
        self.ep_acc_rewards = []
        for k, v in self.train_info.items():
            wandb.log({k: v}, step=total_num_steps)
            
    
    def log_eval(self, total_num_steps):
        self.eval_info['ep_num_act_nodes'] = np.mean(self.ep_num_act_nodes)
        self.eval_info['ep_msg_effi'] = np.mean(self.ep_msg_effi)
        self.eval_info['ep_acc_rewards'] = np.mean(self.ep_acc_rewards)
        self.ep_num_act_nodes = []
        self.ep_msg_effi = []
        self.ep_acc_rewards = []
        for k, v in self.eval_info.items():
            wandb.log({k: v}, step=total_num_steps)


    def run(self):
        episodes = int(self.num_env_steps) // self.episode_length
        terminated = False

        for episode in range(episodes):
            if not terminated:
                obs = self.env.reset()
                states_M = self.policy_net.init_state(1)
            # Initialize the environment and get it's state
            train_episode_reward = 0
            values_worker, values_manager = [], []
            log_probs = []
            rewards, intrinsic_rewards = [], []
            entropies = []  # regularisation
            manager_partial_loss = []
            for step in range(self.episode_length):
                value_worker, value_manager, action_values, goal, tran_grad, states_M = self.policy_net(obs, states_M)
                action_probs = F.softmax(action_values[:, :obs[2].item()], dim=1).to(self.device)
                m = Categorical(probs=action_probs)
                action = m.sample()
                log_prob = m.log_prob(action)
                entropy = -(log_prob * action_probs).sum(1, keepdim=True)
                entropies.append(entropy)
                manager_partial_loss.append(tran_grad)
                
                obs, reward, terminated, edge_feat = self.env.step(action)
                self.IP.pop(0)
                self.IP.append(torch.tensor(edge_feat).to(self.device))
                train_episode_reward += reward
                IP = torch.cat(self.IP, dim=0).to(self.device)
                intrinsic_reward = self.policy_net._intrinsic_reward(IP.unsqueeze(0))
                intrinsic_reward = float(intrinsic_reward)
                values_manager.append(value_manager)
                values_worker.append(value_worker)
                log_probs.append(log_prob)
                rewards.append(reward)
                intrinsic_rewards.append(intrinsic_reward)
                
                if terminated:
                    break
 
            self.ep_num_act_nodes.append(len(self.env.active_nodes))
            self.ep_msg_effi.append(float(self.env.ep_msg_num)/len(self.env.active_nodes))
            self.ep_acc_rewards.append(train_episode_reward)
            
            if terminated:
                obs = self.env.reset()
                states_M = self.policy_net.init_state(1)
                R_worker = torch.zeros(1, 1).to(self.device)
                R_manager = torch.zeros(1, 1).to(self.device)
            # The policy and the value function are updated after every t_max actions or when a terminal state is reaches
            else:
               value_worker, value_manager, _, _, _, _ = self.policy_net(obs, states_M) # Bootstrap from last state
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
                R_worker = self.gamma_worker * R_worker + rewards[i] + self.alpha * intrinsic_rewards[i]
                R_manager = self.gamma_manager * R_manager + rewards[i]
                advantage_worker = R_worker - values_worker[i]
                advantage_manager = R_manager - values_manager[i]
                value_worker_loss = value_worker_loss + 0.5 * advantage_worker.pow(2)
                value_manager_loss = value_manager_loss + 0.5 * advantage_manager.pow(2)

                # Generalized Advantage Estimation
                delta_t_worker = \
                    rewards[i] \
                    + self.alpha * intrinsic_rewards[i]\
                    + self.gamma_worker * values_worker[i + 1].data \
                    - values_worker[i].data
                gae_worker = gae_worker * self.gamma_worker * self.tau_worker + delta_t_worker

                policy_loss = policy_loss \
                    - log_probs[i] * gae_worker - self.entropy_coef * entropies[i]

                if (i + self.policy_net.h) < len(rewards):
                    # TODO try padding the manager_partial_loss with end values (or zeros)
                    manager_loss = manager_loss \
                        - advantage_manager * manager_partial_loss[i + self.policy_net.h]

            self.optimizer.zero_grad()

            total_loss = policy_loss \
                + manager_loss \
                + self.value_manager_loss_coef * value_manager_loss \
                + self.value_worker_loss_coef * value_worker_loss
            self.ep_loss.append(total_loss.item())

            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_num_steps = (episode + 1) * self.episode_length

            if (episode % self.save_interval == 0 or episode == episodes - 1):
                torch.save(self.policy_net.state_dict(), str(self.save_dir) + "/FuN_" + str(episode) + ".pt")

            if episode % self.log_interval == 0:
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}.\n"
                        .format(self.all_args.scenario,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps)) 
                
                self.log_train(total_num_steps)
       
                
    @torch.no_grad()
    def eval(self):
        self.policy_net.load_state_dict(self.model_dir + '/wandb/run20231204/FuN_700.pt')
        self.policy_net.eval()
        terminated = False

        for episode in range(self.eval_episodes):
            if not terminated:
                obs = self.env.reset()
                states_M = self.policy_net.init_state(1)
            # Initialize the environment and get it's state
            eval_episode_reward = 0
            for step in range(self.episode_length):
                _, _, action_values, _, _, states_M = self.policy_net(obs, states_M)
                action_probs = F.softmax(action_values[:, :obs[2].item()], dim=1)
                action = action_probs.max(1, keepdim=True)[1]
                
                obs, reward, terminated, _ = self.env.step(action)
                eval_episode_reward += reward
                
                if terminated:
                    break
 
            self.ep_num_act_nodes.append(len(self.env.active_nodes))
            self.ep_msg_effi.append(float(self.env.ep_msg_num)/len(self.env.active_nodes))
            self.ep_acc_rewards.append(eval_episode_reward)
            
            if terminated:
                self.env.results_recording()
                obs = self.env.reset()
                states_M = self.policy_net.init_state(1)

            total_num_steps = (episode + 1) * self.episode_length
        
        
        self.log_eval(total_num_steps)
        self.env.results_reservation()