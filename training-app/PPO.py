# -*- coding: utf-8 -*-
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

import scipy.signal
from tqdm import tqdm
import torch.nn as nn
from torch.distributions import Categorical
from utils import *


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    """
    A buffer for storing samples
    """

    def __init__(self, fc_feature_size, act_dim, size, gamma=0.99, gae_lambda=0.95, recurrent_size=None,
                 device=torch.device('cpu')):
        self.fc_feature_buf = np.zeros([size, fc_feature_size], dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)  # action
        self.adv_buf = np.zeros(size, dtype=np.float32)  # advantage
        self.rew_buf = np.zeros(size, dtype=np.float32)  # reward
        self.ret_buf = np.zeros(size, dtype=np.float32)  # target value
        self.val_buf = np.zeros(size, dtype=np.float32)  # value
        self.done_buf = np.zeros(size, dtype=np.float32)  # done
        self.logp_buf = np.zeros(size, dtype=np.float32)  # log probability
        # self.mask_buf = np.zeros([size, act_dim], dtype=np.float32)  # mask
        # self.mask_inf_buf = np.zeros([size, act_dim], dtype=np.float32)  # mask
        self.gamma, self.lam = gamma, gae_lambda
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size  # internal index

        if recurrent_size is not None:
            self.mem_buf = np.zeros([size, recurrent_size], dtype=np.float32)
        else:
            self.mem_buf = None

    def store(self, obs, act, rew, val, logp, done, mem=None):
        """
        Store a single transition (state, action, reward, value, log prob) to the buffer
        """
        assert self.ptr < self.max_size
        self.fc_feature_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.done_buf[self.ptr] = done
        self.logp_buf[self.ptr] = logp
        if mem is not None:
            self.mem_buf[self.ptr] = mem
        # self.mask_buf[self.ptr] = mask[0]
        # self.mask_inf_buf[self.ptr] = mask[1]
        self.ptr += 1

    def store_from_other_buffer(self, buffer=None):
        """
        Store th whole episode from another buffer
        """
        if buffer is None:
            return

        assert self.ptr < self.max_size
        self.fc_feature_buf[self.path_start_idx:self.path_start_idx + buffer.ptr] = buffer.fc_feature_buf[:buffer.ptr]
        self.act_buf[self.path_start_idx:self.path_start_idx + buffer.ptr] = buffer.act_buf[:buffer.ptr]
        self.rew_buf[self.path_start_idx:self.path_start_idx + buffer.ptr] = buffer.rew_buf[:buffer.ptr]
        self.val_buf[self.path_start_idx:self.path_start_idx + buffer.ptr] = buffer.val_buf[:buffer.ptr]
        self.logp_buf[self.path_start_idx:self.path_start_idx + buffer.ptr] = buffer.logp_buf[:buffer.ptr]
        self.done_buf[self.path_start_idx:self.path_start_idx + buffer.ptr] = buffer.done_buf[:buffer.ptr]
        # self.mask_buf[self.path_start_idx:self.path_start_idx + buffer.ptr] = buffer.mask_buf[:buffer.ptr]
        # self.mask_inf_buf[self.path_start_idx:self.path_start_idx + buffer.ptr] = buffer.mask_inf_buf[:buffer.ptr]
        if buffer.mem_buf is not None:
            self.mem_buf[self.path_start_idx:self.path_start_idx + buffer.ptr] = buffer.mem_buf[:buffer.ptr]

        self.adv_buf[self.path_start_idx:self.path_start_idx + buffer.ptr] = buffer.adv_buf[:buffer.ptr]
        self.ret_buf[self.path_start_idx:self.path_start_idx + buffer.ptr] = buffer.ret_buf[:buffer.ptr]

        # Update internal index
        self.path_start_idx += buffer.ptr
        self.ptr += buffer.ptr

    def update_delayed_reward(self):
        # update delay reward
        for i in range(0, self.ptr):
            next_idx = i
            founded = False
            while next_idx < self.ptr:
                if int(self.fc_feature_buf[i][6]) == 0:
                    if next_idx == 0 or next_idx == 1 or self.rew_buf[next_idx] > 5000:
                        next_idx += 1
                    else:
                        self.rew_buf[i] = get_true_reward(self.rew_buf[next_idx])
                        founded = True
                        break
                elif int(self.fc_feature_buf[i][6]) == 1 :
                    if next_idx == 1 or self.rew_buf[next_idx] < 5000 or self.rew_buf[next_idx] > 15000 :
                        next_idx += 1
                    else :
                        self.rew_buf[i] = get_true_reward(self.rew_buf[next_idx])
                        founded = True
                        break
                else:
                    if self.rew_buf[next_idx] > 15000:
                        self.rew_buf[i] = get_true_reward(self.rew_buf[next_idx])
                        founded = True
                        break
                    else:
                        next_idx += 1

            if not founded:
                self.ptr = i
                self.done_buf[i-1] = 1.0
                break

    # This function calculate advantage and target value for each episode
    def finish_path(self, last_val=0, last_reward=None):
        if last_reward is not None:
            self.rew_buf[self.ptr - 1] = last_reward

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        td_target = rews[:-1] + self.gamma * vals[1:]
        td_error = td_target - vals[:-1]
        self.adv_buf[path_slice] = td_error
        self.ret_buf[path_slice] = td_target

        # # GAE-Lambda advantage calculation
        # td_target = rews[:-1] + self.gamma * vals[1:]
        # td_error = td_target - vals[:-1]
        # self.adv_buf[path_slice] = discount_cumsum(td_error, self.gamma * self.lam)
        # self.ret_buf[path_slice] = self.adv_buf[path_slice] + vals[:-1]

        # update internal index
        self.path_start_idx = self.ptr

    def get(self, device):
        """
        get data from buffer and reset internal index
        """
        data = dict(fc_feature=self.fc_feature_buf[:self.ptr, :], act=self.act_buf[:self.ptr],
                    # mask=self.mask_buf[:self.ptr, :], mask_inf=self.mask_inf_buf[:self.ptr, :],
                    ret=self.ret_buf[:self.ptr], adv=self.adv_buf[:self.ptr], done=self.done_buf[:self.ptr],
                    logp=self.logp_buf[:self.ptr], mem=self.mem_buf[:self.ptr] if self.mem_buf is not None else 0)

        self.ptr, self.path_start_idx = 0, 0

        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in data.items()}

    def reset(self):
        self.ptr, self.path_start_idx = 0, 0


# Proximal Policy Optimization reinforcement learning algorithm
class PPO(nn.Module):

    def __init__(self,
                 fc_feature_size,
                 act_dim,
                 hidden_size=64,
                 recurrent_size=None,
                 sequence=None,
                 fc_activation=nn.Tanh,
                 buff_length=None,
                 learning_rate=3e-4,
                 train_iters=100,
                 clipping_ratio=0.2,
                 discount=0.99,
                 gae_lambda=0.95,
                 clip_norm=1000,
                 entropy_ratio=0.01,
                 shared_net=0,
                 value_loss_func=loss.MSELoss,
                 optimizer_func=torch.optim.Adam
                 ):
        super().__init__()
        self.fc_feature_size = fc_feature_size
        self.recurrent_size = recurrent_size
        self.sequence = sequence
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.shared_net = shared_net
        self.value_loss_func = value_loss_func

        if shared_net == 1:
            self.fc = nn.Sequential(
                nn.Linear(self.fc_feature_size, hidden_size),
                fc_activation(),
                nn.Linear(hidden_size, hidden_size),
                fc_activation()
            )
            self.embedding_size = hidden_size
        else:
            self.embedding_size = self.fc_feature_size

        if recurrent_size is not None:
            self.recurrent = nn.LSTMCell(self.embedding_size, recurrent_size // 2)
            encoded_dim = recurrent_size // 2
        else:
            self.recurrent = None
            encoded_dim = self.embedding_size

        # Define actor's model
        if shared_net == 1:
            self.actor = nn.Linear(encoded_dim, act_dim)
            self.critic = nn.Linear(encoded_dim, 1)
        else:
            self.actor = nn.Sequential(
                nn.Linear(encoded_dim, hidden_size),
                fc_activation(),
                nn.Linear(hidden_size, hidden_size),
                fc_activation(),
                nn.Linear(hidden_size, act_dim)
            )

            # Define actor's model
            self.critic = nn.Sequential(
                nn.Linear(encoded_dim, hidden_size),
                fc_activation(),
                nn.Linear(hidden_size, hidden_size),
                fc_activation(),
                nn.Linear(hidden_size, 1)
            )

        # Contain data for training at each epoch
        if buff_length is not None:
            self.buf = Buffer(fc_feature_size, act_dim, buff_length, discount, gae_lambda, recurrent_size)

        # Set up optimizers for policy and value function
        self.optimizer = optimizer_func(self.parameters(), lr=learning_rate)
        self.train_iters = train_iters
        self.clip_ratio = clipping_ratio
        self.clip_norm = clip_norm
        self.entropy_ratio = entropy_ratio
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer,
                                                            step_size=1,
                                                            gamma=0.1)

    def _get_pi_and_v(self, fc_feature, memory=None):
        if self.shared_net == 1:
            embedding = self.fc(fc_feature)
        else:
            embedding = fc_feature

        if self.recurrent is not None:
            hidden = (memory[:, :self.recurrent_size // 2], memory[:, self.recurrent_size // 2:])
            hidden = self.recurrent(embedding.reshape(-1, self.embedding_size), hidden)
            encoded_x = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            encoded_x = embedding

        logits = self.actor(encoded_x)

        value = torch.squeeze(self.critic(encoded_x), -1)

        return Categorical(logits=logits), value, memory

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def step(self, obs, memory=None, eval=False, device=torch.device("cpu")):
        fc_feature = torch.tensor(np.array([obs]), dtype=torch.float32).to(device)
        if memory is not None:
            memory = torch.tensor(memory, dtype=torch.float32)
        with torch.no_grad():
            pi, v, new_memory = self._get_pi_and_v(fc_feature, memory)
            if not eval:
                a = pi.sample()
                logp_a = self._log_prob_from_distribution(pi, a)

                return a.numpy(), \
                       v.numpy(), \
                       logp_a.numpy(), \
                       new_memory.numpy() if memory is not None else None
            else:
                a = torch.argmax(pi.probs)
                print(pi.probs)
                return a.numpy(), None, None, \
                       new_memory.numpy() if memory is not None else None

    def act(self, obs, memory=None, device=torch.device("cpu")):
        if memory is not None:
            a, _, _, memory = self.step(obs, memory, True, device)
            return a, memory
        else:
            return self.step(obs, memory, True)[0]

    def update(self, device):
        # Obtain data from Buffer
        data = self.buf.get(device)

        fc_feature, act, adv, logp_old, ret, done, memories = data['fc_feature'], \
                                                              data['act'], \
                                                              data['adv'], \
                                                              data['logp'], \
                                                              data['ret'], \
                                                              data['done'], \
                                                              data['mem']

        # Train policy and value function with multiple steps of gradient descent
        for _ in tqdm(range(self.train_iters)):
            if self.recurrent_size is None:
                self.optimizer.zero_grad()

                # Policy loss
                pi, v, _ = self._get_pi_and_v(fc_feature)
                logp = self._log_prob_from_distribution(pi, act)

                ratio = torch.exp(logp - logp_old)
                clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
                pi_loss = -(torch.min(ratio * adv, clip_adv)).mean()

                # Value loss
                value_loss = self.value_loss_func(v, ret).mean()

                # Entropy loss
                entropy_loss = -torch.sum(pi.probs * torch.log(pi.probs + 1e-20), dim=-1).mean()

                # Loss
                loss = pi_loss + value_loss - self.entropy_ratio * entropy_loss
                loss.backward()
                if self.clip_norm != 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)
                self.optimizer.step()

                return loss.item()
            else:
                total_loss = 0.0
                start_indices = []
                end_indices = np.where(done.numpy() == 1.0)[0]
                start_ii = 0
                for end_ii in end_indices:
                    start_indices.append(np.arange(start_ii, end_ii - self.sequence + 1))
                    start_ii = end_ii + 1

                start_indices = np.concatenate(start_indices)
                memory = memories[start_indices]
                for ii in range(self.sequence):
                    fc_feature_ii, act_ii, adv_ii, logp_old_ii, ret_ii = fc_feature[start_indices + ii], \
                                                                         act[start_indices + ii], \
                                                                         adv[start_indices + ii], \
                                                                         logp_old[start_indices + ii], \
                                                                         ret[start_indices + ii]

                    # Policy loss
                    pi_ii, v_ii, new_memory = self._get_pi_and_v(fc_feature_ii, memory)
                    logp_ii = self._log_prob_from_distribution(pi_ii, act_ii)

                    ratio_ii = torch.exp(logp_ii - logp_old_ii)
                    clip_adv_ii = torch.clamp(ratio_ii, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_ii
                    pi_loss_ii = -(torch.min(ratio_ii * adv_ii, clip_adv_ii)).mean()

                    # Value loss
                    value_loss_ii = self.value_loss_func(v_ii, ret_ii).mean()  # ((v_ii - ret_ii) ** 2).mean()

                    # Entropy loss
                    entropy_loss_ii = -torch.sum(pi_ii.probs * torch.log(pi_ii.probs + 1e-20), dim=-1).mean()

                    # Loss
                    loss_ii = pi_loss_ii + value_loss_ii - self.entropy_ratio * entropy_loss_ii
                    total_loss += loss_ii
                    memory = new_memory

                    if ii < self.sequence:
                        memories[start_indices + ii + 1] = new_memory.detach()

                self.optimizer.zero_grad()
                mean_loss = total_loss / self.sequence
                mean_loss.backward()
                if self.clip_norm != 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)
                self.optimizer.step()

                return mean_loss.item()
