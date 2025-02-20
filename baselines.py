import numpy as np
import os
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils import entropy
from networks import *

'''Baselines for offline RL for discrete space
'''
"------------------------------------Code Block for Behaviour cloning Algorithm--------------------------------------------"

class BehavioralCloning():

    def __init__(self,
                 state_size,
                 action_size,
                 device=None):
        # super(BehavioralCloning, self).__init__(obs_space, action_space, seed)
        self.device = device
        self.lr = 1e-4
        # Number of training iterations
        self.iterations = 0

        # loss function
        self.ce = nn.CrossEntropyLoss()

        # Explicit Policy
        self.actor = Actor(state_size[0],
                            action_size,
                            42
                            ).to(self.device)
        # Optimization
        self.optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=self.lr)

    def get_action(self, state):
        self.actor.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            actions = self.actor(state).cpu()
            action = np.argmax(actions.data.numpy())
        self.actor.train()
        return [action]

    def policy(self, state, eval=False):
        # set networks to eval mode
        self.actor.eval()

        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            actions = self.actor(state).cpu()
            actions = F.softmax(actions, dim=1)
            dist = Categorical(actions.unsqueeze(0))

            return dist.sample().item(), torch.FloatTensor([np.nan]), entropy(actions)

    def train(self, experiences):
        # Sample replay buffer
        state, action, _, _, _ = experiences

        # set networks to train mode
        self.actor.train()

        # predict action the behavioral policy would take
        pred_action = self.actor(state)

        # calculate CE-loss
        loss = self.ce(pred_action, action.squeeze(1))

        # Optimize the policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.iterations += 1

        return torch.mean(loss).detach().cpu().item()


"------------------------------------Code Block for Batch contrained depp Q learning Algorithm--------------------------------------------"
class BCQ():
    """
    Re-implementation of the original author implementation found at
    https://github.com/sfujim/BCQ/blob/master/discrete_BCQ/discrete_BCQ.py
    """

    def __init__(self,
                 state_size,
                 action_size,
                 device=None):

        # epsilon decay
        self.initial_eps = 1.0
        self.lr = 1e-4
        self.device = device
        self.discount = 0.99
        self.end_eps = 1e-2
        self.eps_decay_period = 1000
        self.slope = (self.end_eps - self.initial_eps) / self.eps_decay_period
        self.eval_eps = 0.

        # loss function
        self.huber = nn.SmoothL1Loss()
        self.nll = nn.NLLLoss()

        # threshold for actions, unlikely under the behavioral policy
        self.threshold = 0.3

        # Number of training iterations
        self.iterations = 0

        # After how many training steps 'snap' target to main network?
        self.target_update_freq = 100

        # Q-Networks
        self.Q = DDQN(state_size=state_size,
                            action_size=action_size,
                            layer_size=64
                            ).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        # BCQ has a separate Actor, as I have no common vision network
        self.actor = DDQN(state_size=state_size,
                            action_size=action_size,
                            layer_size=64
                            ).to(self.device)

        # Optimization
        self.optimizer = torch.optim.Adam(params=list(self.Q.parameters())+list(self.actor.parameters()), lr=self.lr)

    def get_action(self, state):
        self.actor.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            actions = self.actor(state).cpu()
            action = np.argmax(actions.data.numpy())
        self.actor.train()
        return [action]

    def policy(self, state, eval=False):

        # set networks to eval mode
        self.actor.eval()
        self.Q.eval()

        if eval:
            eps = self.eval_eps
        else:
            eps = max(self.slope * self.iterations + self.initial_eps, self.end_eps)

        # epsilon greedy policy
        if self.rng.uniform(0, 1) > eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)

                q_val = self.Q(state).cpu()
                actions = self.actor(state).cpu()

                sm = F.log_softmax(actions, dim=1).exp()
                mask = ((sm / sm.max(1, keepdim=True)[0]) > self.threshold).float()

                # masking non-eligible values with -9e9 to be sure they are not sampled
                return int((mask * q_val + (1. - mask) * -9e9).argmax(dim=1)), \
                       q_val, entropy(sm)
        else:
            return self.rng.integers(self.action_space), np.nan, np.nan

    def train(self, experiences):
        # Sample replay buffer
        state, action, reward, next_state, not_done = experiences

        # set networks to train mode
        self.actor.train()
        self.Q.train()
        self.Q_target.train()

        with torch.no_grad():
            q_val = self.Q(state)
            actions = self.actor(state)

            sm = F.log_softmax(actions, dim=1).exp()
            sm = (sm / sm.max(1, keepdim=True)[0] > self.threshold).float()
            next_action = (sm * q_val + (1. - sm) * -9e9).argmax(dim=1, keepdim=True)

            q_val = self.Q_target(state)
            target_Q = reward + not_done * self.discount * q_val.gather(1, next_action).reshape(-1, 1)

        # Get current Q estimate and actor decisions on actions
        current_Q = self.Q(state).gather(1, action)
        actions = self.actor(state)

        # Compute Q loss (Huber loss)
        Q_loss = self.huber(current_Q, target_Q)
        A_loss = self.nll(F.log_softmax(actions, dim=1), action.reshape(-1))
        # third term is
        loss = Q_loss + A_loss + 1e-2 * actions.pow(2).mean()

        # Optimize the Q
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.iterations += 1
        # Update target network by full copy every X iterations.
        if self.iterations % self.target_update_freq == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

        return torch.mean(Q_loss).detach().cpu().item()

    def get_name(self) -> str:
        return "BatchConstrainedQLearning"

"------------------------------------Code Block for Behavior Value Estimation Algorithm--------------------------------------------"
class BVE():

    def __init__(self,
                 state_size,
                 action_size,
                 device=None):
        # super(BVE, self).__init__(obs_space, action_space, discount, lr, seed)

        # epsilon decay
        self.initial_eps = 1.0
        self.lr = 1e-4
        self.device = device
        self.discount = 0.99
        self.end_eps = 1e-2
        self.eps_decay_period = 1000
        self.slope = (self.end_eps - self.initial_eps) / self.eps_decay_period
        self.eval_eps = 0.

        # loss function
        self.huber = nn.SmoothL1Loss()

        # Number of training iterations
        self.iterations = 0

        # After how many training steps 'snap' target to main network?
        self.target_update_freq = 100

        # Q-Networks
        self.actor = DDQN(state_size=state_size,
                            action_size=action_size,
                            layer_size=64
                            ).to(self.device)

        # Optimization
        self.optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=self.lr)

    def get_action(self, state):
        self.actor.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            actions = self.actor(state).cpu()
            action = np.argmax(actions.data.numpy())
        self.actor.train()
        return [action]

    def policy(self, state, eval=False):

        # set networks to eval mode
        self.actor.eval()

        if eval:
            eps = self.eval_eps
        else:
            eps = max(self.slope * self.iterations + self.initial_eps, self.end_eps)

        # epsilon greedy policy
        if self.rng.uniform(0, 1) > eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_val = self.actor(state).cpu()
                return q_val.argmax().item(), q_val, np.nan
        else:
            return self.rng.integers(self.action_space), np.nan, np.nan

    def train(self, experiences):
        # Sample replay buffer
        state, action, reward, next_state, not_done = experiences

        # set networks to train mode
        self.actor.train()

        # Compute the target Q value
        with torch.no_grad():
            target_Q = reward + not_done * self.discount * self.actor(next_state).gather(1, action)

        # Get current Q estimate
        current_Q = self.actor(state).gather(1, action)

        # Compute Q loss (Huber loss)
        Q_loss = self.huber(current_Q, target_Q)

        # Optimize the Q
        self.optimizer.zero_grad()
        Q_loss.backward()
        self.optimizer.step()

        self.iterations += 1

        return torch.mean(Q_loss).detach().cpu().item()

    def get_name(self) -> str:
        return "BehavioralValueEstimation"


"------------------------------------Code Block for Constrained Q Learning Algorithm--------------------------------------------"
class CQL():

    def __init__(self,
                 state_size,
                 action_size,
                 device=None):
        # super(CQL, self).__init__(obs_space, action_space, discount, lr, seed)

        # epsilon decay
        self.initial_eps = 1.0
        self.end_eps = 1e-2
        self.eps_decay_period = 1000
        self.lr = 1e-4
        self.device = device
        self.discount = 0.99
        self.slope = (self.end_eps - self.initial_eps) / self.eps_decay_period
        self.eval_eps = 0.

        # loss function
        self.huber = nn.SmoothL1Loss()
        self.ce = nn.CrossEntropyLoss()

        # Number of training iterations
        self.iterations = 0

        # After how many training steps 'snap' target to main network?
        self.target_update_freq = 100

        # Q-Networks
        self.Q = Critic(state_size[0], action_size, 42).to(self.device)

        self.Q_target = copy.deepcopy(self.Q)

        # Optimization
        self.optimizer = torch.optim.Adam(params=self.Q.parameters(), lr=self.lr)

        # temperature parameter
        self.alpha = 0.1

    def get_action(self, state):
        self.Q.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            actions = self.Q(state).cpu()
            action = np.argmax(actions.data.numpy())
        self.Q.train()
        return [action]
    
    def policy(self, state, eval=False):

        # set networks to eval mode
        self.Q.eval()

        if eval:
            eps = self.eval_eps
        else:
            eps = max(self.slope * self.iterations + self.initial_eps, self.end_eps)

        # epsilon greedy policy
        if self.rng.uniform(0, 1) > eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_val = self.Q(state).cpu()
                return q_val.argmax().item(), q_val, np.nan
        else:
            return self.rng.integers(self.action_space), np.nan, np.nan

    def train(self, experiences):
        # Sample replay buffer
        state, action, reward, next_state, not_done = experiences

        # set networks to train mode
        self.Q.train()
        self.Q_target.train()

        ### Train main network

        # Compute the target Q value
        with torch.no_grad():
            q_val = self.Q(next_state)
            next_action = q_val.argmax(dim=1, keepdim=True)
            target_Q = reward + not_done * self.discount * self.Q_target(next_state).gather(1, next_action)

        # Get current Q estimate
        current_Qs = self.Q(state)
        current_Q = current_Qs.gather(1, action)

        # Compute Q loss (Huber loss)
        Q_loss = self.huber(current_Q, target_Q)

        # calculate regularizing loss
        R_loss = torch.mean(self.alpha * (torch.logsumexp(current_Qs, dim=1) - current_Qs.gather(1, action).squeeze(1)))

        # Optimize the Q
        self.optimizer.zero_grad()
        (Q_loss + R_loss).backward()
        self.optimizer.step()

        self.iterations += 1
        # Update target network by full copy every X iterations.
        if self.iterations % self.target_update_freq == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

        return torch.mean(Q_loss).detach().cpu().item()

    def get_name(self) -> str:
        return "ConservativeQLearning"


"------------------------------------Code Block for Critic Regularized Regression Algorithm--------------------------------------------"
class CRR():

    def __init__(self,
                 state_size,
                 action_size,
                 device=None):
        
        #super(CRR, self).__init__(obs_space, action_space, discount, lr, seed)

        # epsilon decay
        self.initial_eps = 1.0
        self.end_eps = 1e-2
        self.eps_decay_period = 1000
        self.lr = 1e-4
        self.device = device
        self.discount = 0.99
        self.slope = (self.end_eps - self.initial_eps) / self.eps_decay_period
        self.eval_eps = 0.

        # Number of training iterations
        self.iterations = 0

        # After how many training steps 'snap' target to main network?
        self.target_update_freq = 100

        # Q-Networks
        self.Q = DDQN(state_size=state_size,
                            action_size=action_size,
                            layer_size=64
                            ).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)

        # policy network
        self.actor = DDQN(state_size=state_size,
                            action_size=action_size,
                            layer_size=64
                            ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        self.ce = nn.CrossEntropyLoss(reduction='none')

        # huber loss
        self.huber = nn.SmoothL1Loss()

        # Optimization
        self.optimizer = torch.optim.Adam(params=self.Q.parameters(), lr=self.lr)
        self.p_optim = torch.optim.Adam(params=self.actor.parameters(), lr=self.lr)

        # Temperature parameter
        self.beta = 1
        # parameter for advantage estimate
        self.m = 4

    def get_action(self, state):
        self.actor.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            actions = self.actor(state).cpu()
            action = np.argmax(actions.data.numpy())
        self.actor.train()
        return [action]

    def policy(self, state, eval=False):

        # set networks to eval mode
        self.actor.eval()
        self.Q.eval()

        if eval:
            eps = self.eval_eps
        else:
            eps = max(self.slope * self.iterations + self.initial_eps, self.end_eps)

        # epsilon greedy policy
        if self.rng.uniform(0, 1) > eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_val = self.Q(state).cpu()

                actions = self.actor(state).cpu()
                actions = F.softmax(actions, dim=1)
                dist = Categorical(actions.unsqueeze(0))

                return dist.sample().item(), q_val, entropy(actions)
        else:
            return self.rng.integers(self.action_space), np.nan, np.nan

    def train(self, experiences):
        # Sample replay buffer
        state, action, reward, next_state, not_done = experiences

        ###################################
        ### Policy update
        ###################################

        # set networks to train mode
        self.actor.train()
        self.actor_target.train()
        self.Q.train()
        self.Q_target.train()

        # calculate advantage
        with torch.no_grad():
            current_Qs = self.Q(state)
            baseline = []
            # sample m times for baseline
            for _ in range(self.m):
                actions = self.actor(state)
                probs = F.softmax(actions, dim=1)
                dist = Categorical(probs)
                baseline.append(current_Qs.gather(1, dist.sample().unsqueeze(1)))
            baseline = torch.stack(baseline, dim=0)
            # mean style
            advantage = current_Qs - torch.mean(baseline, dim=0)
            # max style
            #advantage = current_Qs - torch.max(baseline, dim=0)[0]

        # policy loss
        # exp style
        #loss = (self.ce(self.actor(state), action.squeeze(1)).unsqueeze(1) *
        #        torch.exp(advantage / self.beta).gather(1, action)).mean()
        # binary style
        loss = (self.ce(self.actor(state), action.squeeze(1)).unsqueeze(1) *
                torch.heaviside(advantage, values=torch.zeros(1).to(self.device)).gather(1, action)).mean()

        # optimize policy
        self.p_optim.zero_grad()
        loss.backward()
        self.p_optim.step()

        ###################################
        ### Critic update
        ###################################

        # set networks to train mode
        self.actor.train()
        self.actor_target.train()
        self.Q.train()
        self.Q_target.train()

        # Compute the target Q value
        with torch.no_grad():
            actions = self.actor_target(next_state)
            probs = F.softmax(actions, dim=1)
            dist = Categorical(probs)
            target_Q = reward + not_done * self.discount * self.Q_target(next_state).gather(1, dist.sample().unsqueeze(1))

        # Get current Q estimate
        current_Q = self.Q(state).gather(1, action)

        # Compute Q loss (Huber loss)
        Q_loss = self.huber(current_Q, target_Q)

        # Optimize the Q
        self.optimizer.zero_grad()
        Q_loss.backward()
        self.optimizer.step()

        self.iterations += 1
        # Update target network by full copy every X iterations.
        if self.iterations % self.target_update_freq == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())
            self.actor_target.load_state_dict(self.actor.state_dict())

        return torch.mean(Q_loss)

    def get_name(self) -> str:
        return "CriticRegularizedRegression"

"------------------------------------Code Block for Monte Carlo Regularization Algorithm--------------------------------------------"
class MCE():

    def __init__(self,
                 state_size,
                 action_size,
                 device=None):
        # super(MCE, self).__init__(obs_space, action_space, discount, lr, seed)

        self.eval_eps = 0.
        self.lr = 1e-4
        self.device = device
        self.discount = 0.99

        # loss function
        self.huber = nn.SmoothL1Loss()

        # Number of training iterations
        self.iterations = 0

        # Q-Networks
        self.actor = DDQN(state_size=state_size,
                            action_size=action_size,
                            layer_size=64
                            ).to(self.device)

        # Optimization
        self.optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=self.lr)

    def get_action(self, state):
        self.actor.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            actions = self.actor(state).cpu()
            action = np.argmax(actions.data.numpy())
        self.actor.train()
        return [action]

    def policy(self, state, eval=False):

        # set network to eval mode
        self.actor.eval()

        # epsilon greedy policy
        if self.rng.uniform(0, 1) > self.eval_eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_val = self.actor(state).cpu()
                return q_val.argmax().item(), q_val, np.nan
        else:
            return self.rng.integers(self.action_space), np.nan, np.nan

    def train(self, experiences):
        # Sample replay buffer
        state, action, reward, next_state, not_done = experiences

        # set network to train mode
        self.actor.train()

        # Get current Q estimate
        current_Q = self.actor(state).gather(1, action)

        # Compute Q loss (Huber loss)
        Q_loss = self.huber(current_Q, reward)

        # Optimize the Q
        self.optimizer.zero_grad()
        Q_loss.backward()
        self.optimizer.step()

        self.iterations += 1

        return torch.mean(Q_loss).detach().cpu().item()

    def get_name(self) -> str:
        return "Monte-Carlo Estimation"

"------------------------------------Code Block for Random Ensemble Mixture Algorithm--------------------------------------------"
class REM():

    def __init__(self,
                 state_size,
                 action_size,
                 device=None):
        # super(REM, self).__init__(obs_space, action_space, discount, lr, seed)

        # epsilon decay
        self.initial_eps = 1.0
        self.end_eps = 1e-2
        self.lr = 1e-4
        self.device = device
        self.action_space = action_size
        self.discount = 0.99
        self.eps_decay_period = 1000
        self.slope = (self.end_eps - self.initial_eps) / self.eps_decay_period
        self.eval_eps = 0

        # loss function
        self.huber = nn.SmoothL1Loss()

        # Number of training iterations
        self.iterations = 0

        # After how many training steps 'snap' target to main network?
        self.target_update_freq = 100

        # Q-Networks
        self.actor = RemCritic(state_size[0], action_size, 42, 200).to(self.device)
        self.Q_target = copy.deepcopy(self.actor)

        # Optimization
        self.optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=self.lr)

    def get_action(self, state):
        self.actor.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            actions = self.actor(state).cpu()
            action = np.argmax(actions.data.numpy())
        self.actor.train()
        return [action]

    def policy(self, state, eval=False):

        # set networks to eval mode
        self.actor.eval()

        if eval:
            eps = self.eval_eps
        else:
            eps = max(self.slope * self.iterations + self.initial_eps, self.end_eps)

        # epsilon greedy policy
        if self.rng.uniform(0, 1) > eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_val = self.actor(state).cpu()
                return q_val.argmax().item(), q_val, np.nan
        else:
            return self.rng.integers(self.action_space), np.nan, np.nan

    def train(self, experiences):
        # Sample replay buffer
        state, action, reward, next_state, not_done = experiences

        # set networks to train mode
        self.actor.train()
        self.Q_target.train()

        # Compute the target Q value
        with torch.no_grad():
            q_val = self.actor(next_state)
            next_action = q_val.argmax(dim=1, keepdim=True)
            target_Q = reward + not_done * self.discount * self.Q_target(next_state).gather(1, next_action)

        # Get current Q estimate
        current_Q = self.actor(state).gather(1, action)

        # Compute Q loss (Huber loss)
        Q_loss = self.huber(current_Q, target_Q)

        # Optimize the Q
        self.optimizer.zero_grad()
        Q_loss.backward()
        self.optimizer.step()

        self.iterations += 1
        # Update target network by full copy every X iterations.
        if self.iterations % self.target_update_freq == 0:
            self.Q_target.load_state_dict(self.actor.state_dict())

        return  torch.mean(Q_loss).detach().cpu().item()

    def get_name(self) -> str:
        return "RandomEnsembleMixture"


"------------------------------------Code Block for Random Ensemble Mixture Algorithm--------------------------------------------"
class QRDQN():

    def __init__(self,
                 state_size,
                 action_size,
                 device=None):
        # super(QRDQN, self).__init__(obs_space, action_space, discount, lr, seed)

        # epsilon decay
        self.initial_eps = 1.0
        self.end_eps = 1e-2
        self.eps_decay_period = 1000
        self.lr = 1e-4
        self.discount = 0.99
        self.quantiles = 50
        self.device = device
        self.obs_space = state_size
        self.action_space = action_size
        self.slope = (self.end_eps - self.initial_eps) / self.eps_decay_period
        self.eval_eps = 0.

        # Number of training iterations
        self.iterations = 0

        # After how many training steps 'snap' target to main network?
        self.target_update_freq = 100

        # Quantiles
        self.quantile_tau = torch.FloatTensor([i / self.quantiles for i in range(1, self.quantiles + 1)]).to(self.device)

        # Q-Networks
        self.actor = QrCritic(self.obs_space[0], self.action_space, 42, self.quantiles).to(self.device)
        self.Q_target = copy.deepcopy(self.actor)

        # huber loss
        self.huber = nn.SmoothL1Loss(reduction='none')

        # Optimization
        self.optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=self.lr)

    def get_action(self, state):
        self.actor.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            actions = self.actor(state).cpu()
            action = np.argmax(actions.data.numpy())
        self.actor.train()
        return [action]

    def policy(self, state, eval=False):

        # set networks to eval mode
        self.actor.eval()

        if eval:
            eps = self.eval_eps
        else:
            eps = max(self.slope * self.iterations + self.initial_eps, self.end_eps)

        # epsilon greedy policy
        if self.rng.uniform(0, 1) > eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_val = self.actor(state).cpu()
                return q_val.argmax().item(), q_val, np.nan
        else:
            return self.rng.integers(self.action_space), np.nan, np.nan

    def train(self, experiences):
        # Sample replay buffer
        state, action, reward, next_state, not_done = experiences

        # set networks to train mode
        self.actor.train()
        self.Q_target.train()

        # Compute the target Q value
        with torch.no_grad():
            target_Qs = self.actor(next_state)
            action_indices = torch.argmax(target_Qs.mean(dim=2), dim=1, keepdim=True)
            target_Qs = self.Q_target(next_state)
            target_Qs = target_Qs.gather(1, action_indices.unsqueeze(2).expand(-1, 1, self.quantiles))
            target_Qs = reward.unsqueeze(1) + not_done.unsqueeze(1) * self.discount * target_Qs

        # Get current Q estimate
        current_Qs = self.actor(state).gather(1, action.unsqueeze(2).expand(-1, 1,self.quantiles)).transpose(1, 2)

        # expand along singular dimensions
        target_Qs = target_Qs.expand(-1, self.quantiles, self.quantiles)
        current_Qs = current_Qs.expand(-1, self.quantiles, self.quantiles)

        # Compute TD error
        td_error = target_Qs - current_Qs

        # calculate loss through TD
        huber_l = self.huber(current_Qs, target_Qs)

        # calculate quantile loss
        quantile_loss = abs(self.quantile_tau - (td_error.detach() < 0).float()) * huber_l
        quantile_loss = quantile_loss.sum(dim=1).mean(dim=1).mean()

        # Optimize the Q
        self.optimizer.zero_grad()
        quantile_loss.backward()
        self.optimizer.step()

        self.iterations += 1
        # Update target network by full copy every X iterations.
        if self.iterations % self.target_update_freq == 0:
            self.Q_target.load_state_dict(self.actor
                                          .state_dict())

        return torch.mean(quantile_loss).detach().cpu().item()

    def get_name(self) -> str:
        return "QuantileRegressionDeepQNetwork"

    
