import time
from networkx import selfloop_edges
import numpy as np
import math
import random
from functools import reduce
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable
import wandb
from Algorithm import FeudalNet as Policy
from ExperienceReplay import Transition, ReplayMemory
from itertools import count

def _t2n(x):
    return x.detach().cpu().numpy()

class DQNRunner(object):
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
        self.hidden_dim = self.all_args.hidden_dim
        self.use_wandb = self.all_args.use_wandb
        self.TAU = self.all_args.tau

        # log
        self.ep_loss = []
        self.train_info = {}
        self.train_info['ep_aver_loss'] = 0
        self.train_info['aver_step_reward'] = 0
        self.train_info['ep_acc_rewards'] = 0
        self.train_info['ep_num_act_nodes'] = 0
        self.train_info['ep_msg_effi'] = 0
        
        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        self.save_dir = str(wandb.run.dir)
        self.run_dir = str(wandb.run.dir)

        # policy network
        self.policy_net = Policy(self.env.num_feats,
                                 self.hidden_dim)

        self.target_net = Policy(self.env.num_feats,
                                 self.hidden_dim)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.all_args.lr, amsgrad=True)

        if self.model_dir is not None:
            self.restore(self.model_dir)

        self.buffer = ReplayMemory(self.all_args.memory_size)


    def select_action(self, state, step):
        sample = random.random()
        eps_threshold = self.all_args.eps_end + (self.all_args.eps_start - self.all_args.eps_end) * math.exp(-1. * step / self.all_args.eps_decay)
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state)[0][:state[2].item()].view(1, state[2].item()).max(1)[1].view(1, 1)
        else:
            return torch.tensor(random.randint(0, state[2].item() - 1)).view(1, 1)


    def log_train(self, total_num_steps):
        iter_buffer = iter(self.buffer.memory)
        rewards = [Transition.reward.item() for Transition in iter_buffer]
        self.train_info['aver_step_reward'] = np.mean(rewards)
        self.train_info['ep_aver_loss'] = np.mean(self.ep_loss)
        self.ep_loss = []
        print('average_step_reward is {}.'.format(self.train_info['aver_step_reward']))
        for k, v in self.train_info.items():
            wandb.log({k: v}, step=total_num_steps)


    def optimize_model(self):
        if len(self.buffer) < self.all_args.batch_size:
            return
        transitions = self.buffer.sample(self.all_args.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = tuple([s for s in batch.next_state if s is not None])
        
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(batch.state).gather(1, action_batch)

        next_state_values = torch.zeros(self.all_args.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
    
        expected_state_action_values = (next_state_values * self.all_args.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.ep_loss.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def run(self):
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length

        for episode in range(episodes):
            # Initialize the environment and get it's state
            state = self.env.reset()
            train_episode_reward = 0
            for step in range(self.episode_length):
                action = self.select_action(state, step).to(self.device)
                observation, reward, terminated = self.env.step(action)
                train_episode_reward += reward
                reward = torch.tensor([reward], device=self.device)
                done = terminated

                if terminated:
                    next_state = None
                    self.train_info['ep_num_act_nodes'] = len(self.env.active_nodes)
                    self.train_info['ep_msg_effi'] = float(self.env.ep_msg_num)/len(self.env.active_nodes)
                    self.train_info['ep_acc_rewards'] = train_episode_reward
                else:
                    next_state = observation

                # Store the transition in memory
                self.buffer.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    break

            total_num_steps = (episode + 1) * self.episode_length

            if (episode % self.save_interval == 0 or episode == episodes - 1):
                torch.save(self.policy_net.state_dict(), str(self.save_dir) + "/DQN_" + str(episode) + ".pt")

            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}.\n"
                        .format(self.all_args.scenario,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps)) 
                
                self.log_train(total_num_steps)


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
        self.optimizer = self.all_args.optimizer

        # log
        self.ep_loss = []
        self.train_info = {}
        self.train_info['ep_aver_loss'] = 0
        self.train_info['ep_acc_rewards'] = 0
        self.train_info['ep_num_act_nodes'] = 0
        self.train_info['ep_msg_effi'] = 0
        
        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        self.save_dir = str(wandb.run.dir)
        self.run_dir = str(wandb.run.dir)

        # Feudal network
        self.policy_net = Policy(self.env.num_feats,
                                 self.hidden_dim)
        self.policy_net.share_memory()
        self.IP = [torch.zeros(self.policy_net.k, requires_grad=False) for _ in range(self.policy_net.h)]
        
        if self.optimizer is None:
            print("no shared optimizer")
            optimizer = optim.Adam(self.policy_net.parameters(), lr=self.all_args.lr)

        if self.model_dir is not None:
            self.restore(self.model_dir)


    def log_train(self, total_num_steps):
        self.train_info['ep_aver_loss'] = np.mean(self.ep_loss)
        self.ep_loss = []
        for k, v in self.train_info.items():
            wandb.log({k: v}, step=total_num_steps)


    def run(self):
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length

        for episode in range(episodes):
            # Initialize the environment and get it's state
            obs = self.env.reset()
            states_M = self.policy_net.init_state(1)
            train_episode_reward = 0
            values_worker, values_manager = [], []
            log_probs = []
            rewards, intrinsic_rewards = [], []
            entropies = []  # regularisation
            manager_partial_loss = []
            for step in range(self.episode_length):
                value_worker, value_manager, action_probs, goal, tran_grad, states_M = self.policy_net(obs, states_M)
                m = Categorical(probs=action_probs)
                action = m.sample()
                log_prob = m.log_prob(action)
                entropy = -(log_prob * action_probs).sum(1, keepdim=True)
                entropies.append(entropy)
                manager_partial_loss.append(tran_grad)
                
                obs, reward, terminated, edge_feat = self.env.step(action)
                self.IP.pop(0)
                self.IP.append(edge_feat)
                train_episode_reward += reward
                intrinsic_reward = self.policy_net._intrinsic_reward(self.IP)
                intrinsic_reward = float(intrinsic_reward)
                values_manager.append(value_manager)
                values_worker.append(value_worker)
                log_probs.append(log_prob)
                rewards.append(reward)
                intrinsic_rewards.append(intrinsic_reward)
                done = terminated

                if terminated:
                    next_state = None
                    states_M = self.policy_net.init_state(1)
                    
                    self.train_info['ep_num_act_nodes'] = len(self.env.active_nodes)
                    self.train_info['ep_msg_effi'] = float(self.env.ep_msg_num)/len(self.env.active_nodes)
                    self.train_info['ep_acc_rewards'] = train_episode_reward
                else:
                    next_state = obs

                # Move to the next state
                state = next_state
                
                if done:
                    break
             
            # The policy and the value function are updated after every t_max actions or when a terminal state is reached
            R_worker = torch.zeros(1, 1)
            R_manager = torch.zeros(1, 1)
            if not done:
               value_worker, value_manager, _, _, _, _ = self.policy_net(state, states_M) # Bootstrap from last state
               R_worker = value_worker.data
               R_manager = value_manager.data
                
            values_worker.append(Variable(R_worker))
            values_manager.append(Variable(R_manager))
            policy_loss = 0
            manager_loss = 0
            value_manager_loss = 0
            value_worker_loss = 0
            gae_worker = torch.zeros(1, 1)
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
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}.\n"
                        .format(self.all_args.scenario,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps)) 
                
                self.log_train(total_num_steps)   
                