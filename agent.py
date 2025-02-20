import torch
import torch.nn as nn
from networks import DDQN, Critic
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import random
from teacher import *
from torch.distributions import Categorical
import copy

class CQLAgent():
    def __init__(self, state_size, action_size, hidden_size=64, device="cpu", config=None):
        self.state_size = state_size
        self.action_size = action_size
        self.eps_rule = 1.0
        self.device = device
        self.tau = 1e-3
        self.seed = 42
        self.gamma = 0.99

        # parameters for configuration file
        self.env_name = config.env
        self.lam = config.lam
        self.lr = config.lr
        self.use_teacher = config.use_teach
        self.warm_start = config.warm_start
        self.teacher_update = config.teacher_update
        self.behaviour_policy = None # PPO.load("/home/ubuntu/OODOfflineRL/ppo-LunarLander-discrete-v2")
        
        
        self.student_network = DDQN(state_size=self.state_size,
                            action_size=self.action_size,
                            layer_size=hidden_size
                            ).to(self.device)

        self.target_net = DDQN(state_size=self.state_size,
                            action_size=self.action_size,
                            layer_size=hidden_size
                            ).to(self.device)
        
        self.optimizer = optim.Adam(params=self.student_network.parameters(), lr=self.lr) #default 1e-4

        
        # Teacher initialization for cartpole env
        if 'Cart' in self.env_name:
            self.teacher_actor = init_cart_nets('one_hot')
        # Teacher initialization for lunar lander env
        if 'Lunar' in self.env_name:
            self.teacher_actor = init_lander_nets('one_hot')
        if 'Mountain' in self.env_name:
            self.teacher_actor = init_mountain_nets('one_hot')

            # Initialize optimizer for teacher critic will use this for teacher training later
        if self.use_teacher:
            self.teacher_optimizer = optim.Adam(self.teacher_actor.parameters(), lr=1e-5)

    #Function to check the entropy of the actions predicted
    def calculate_entropy(self, predictions):
        # Convert predictions to probabilities using softmax
        probs = torch.nn.functional.softmax(predictions, dim=-1)
        # Calculate entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        return entropy.mean()

     #Function to calculate the uncertainity of the states over 100 forward passes
    def calculate_uncertainity(self,action_rule,action_pred,states_mismatch):
        num_forward_passes = 10
        # self.student_network.eval()
        # Lists to store predictions
        predictions_pred = []
        predictions_rule = []
        for i in range(num_forward_passes):
            Q_val = self.student_network(states_mismatch, training=False).detach()
            # Gather the Q values for rule action and student action
            Q_val_pred = Q_val.gather(1, action_pred)
            Q_val_rule = Q_val.gather(1, action_rule)
            # Append predictions to lists
            predictions_pred.append(Q_val_pred.cpu().numpy())
            predictions_rule.append(Q_val_rule.cpu().numpy())
        # Convert predictions to tensors
        predictions_pred = torch.tensor(predictions_pred)
        predictions_rule = torch.tensor(predictions_rule)
        # Calculate the variance for predictions over 10 forward passes
        variance_pred = torch.var(predictions_pred, dim=0)
        variance_rule = torch.var(predictions_rule, dim=0)
        # self.student_network.train()
        return torch.mean(variance_pred), torch.mean(variance_rule)


    # Get action from the teacher network
    # The network return a probability distribution over the action
    def get_action_teacher(self, observation):
        obs = torch.Tensor(observation)
        self.teacher_actor.eval()
        if 'Cart' in self.env_name:
            obs = obs.view(1, -1)
        probs = self.teacher_actor(obs)
        if 'Cart' in self.env_name:
            probs = probs.view(-1)
        action = np.argmax(probs.cpu().data.numpy())
        self.teacher_actor.train()
        return action, probs.cpu()  
    
    def get_action_teacher_net(self, observation):
        obs = torch.Tensor(observation)
        self.teacher_actor.eval()
        with torch.no_grad():
            action_values = self.teacher_actor(obs)
        self.teacher_actor.train()
        return np.argmax(action_values.cpu().data.numpy()), action_values.cpu()

        
    # Get the determinstic action from the network
    def get_action(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.student_network.eval()
            with torch.no_grad():
                action_values = self.student_network(state)
            self.student_network.train()
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
        else:
            action = random.choices(np.arange(self.action_size), k=1)
        return action

    def get_deter_action(self, state):
        # state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.student_network.eval()
        with torch.no_grad():
            action_values = self.student_network(state.unsqueeze(0))
            # print(f'state ----- {state} ----action_values--- {action_values}')
        self.student_network.train()
        action = np.argmax(action_values.cpu().data.numpy(), axis=1)
        return action
    

    # For now let us see if having a behaviour policy helps
    # Replace actions for each step of behaviour policy
    def calculate_rule_loss_behaviour(self, states):
        # Let us only consider the simplest rule right now 
        # Obs[1] < 0.06 ∧ Obs[6] == 1 ∧ Obs[7] == 1 -> action == 0
        accumulated_penalty = 0
        for s in states:
            predicted_action = self.get_deter_action(s)
            state = np.array([s.cpu().data.numpy()])
            #selct actions using PPO algorithm
            actions, states = self.behaviour_policy.predict(
                state,  # type: ignore[arg-type]
                state=None,
                episode_start=1,
                deterministic=True,
            )
            rule_action = actions[0]

            # Calculate euclidean distance between both actions 
            dist = np.linalg.norm(predicted_action - rule_action)
            accumulated_penalty += dist

        return accumulated_penalty

    # This is main function for integration of teacher loss with the student network
    # Reduce the Q value for the stundent action
    # Increase the Q value for the teacher predicted action
    def calculate_teacher_loss(self, states, Q_expected, ep):
        action_rule = []
        action_pred = []
        states_mismatch = []
        rule_error = 0
        # print(f'Q_expected ======= {Q_expected}')
        state_copy = copy.deepcopy(states)
        for s in state_copy:
            condition = check_condition(s,self.env_name)
            # Triggering in conditions for which the rules are written for, specific to environment
            if (condition):
                # Get the predicted action from the oriinal student_network
                predicted_action = self.get_deter_action(s)

                # selct actions using the rule based DDT
                rule_action, prob = self.get_action_teacher(s)

                # Increase Q value for rule action and decrease Q value for predicted action if there is an action mismatch
                # If actions dont match for a particular state add it to the list of states
                if(predicted_action!=rule_action):
                    # print(f'state mismatch ======= {s.cpu().data.numpy()} ========== {rule_action} ========= {predicted_action}')
                    states_mismatch.append(s.cpu().data.numpy())             
                    action_rule.append(rule_action)
                    action_pred.append(predicted_action[0])

        if len(states_mismatch)!=0:
            action_rule = torch.from_numpy(np.vstack([a for a in action_rule])).long().to(self.device)
            action_pred = torch.from_numpy(np.vstack([a for a in action_pred])).long().to(self.device)
            states_mismatch = torch.from_numpy(np.vstack([s for s in states_mismatch])).float().to(self.device)
            # print(f'action_pred ======= {action_pred}')

            # Calculate Q values for only the states with mismatched action
            Q_val = self.student_network(states_mismatch)
            
            # Gathe the Q values for rule action and student action
            Q_val_pred = Q_val.gather(1,action_pred)
            Q_val_rule = Q_val.gather(1,action_rule)
            
            # Shift the Q value of student action to teacher action
            if ep<self.warm_start:
                rule_error = F.mse_loss(Q_val_pred, Q_val_rule) + F.mse_loss(Q_val_rule, Q_val_pred)
            else:
                # Do not update the student network in this case and train the teacher network
                rule_error = 0
                '''if ep%self.teacher_update ==0 and torch.mean(Q_val_rule) < torch.mean(Q_val_pred):
                    var_student, var_teacher = self.calculate_uncertainity(action_rule,action_pred,states_mismatch)
                    if var_student < var_teacher:
                        teacher_loss = self.teacher_learn(states)
                        print(f'--------Training teacher-------------{teacher_loss}----------{var_student}------- {var_teacher}')
                        rule_error = F.mse_loss(Q_expected, Q_expected)
                    else:
                        rule_error = F.mse_loss(Q_val_pred, Q_val_rule) + F.mse_loss(Q_val_rule, Q_val_pred)
                else:
                    rule_error = F.mse_loss(Q_val_pred, Q_val_rule) + F.mse_loss(Q_val_rule, Q_val_pred)'''
        else:
            # else set the rule error to be 0
            rule_error = F.mse_loss(Q_expected, Q_expected)
        # print(rule_error)
        
        return rule_error

    # Code for updating the teacher network based on uncertainity
    # Uses cross entropy loss between the student logits and the teacher logits
    def teacher_learn(self,experiences):
        states = experiences
        logits_teacher = self.teacher_actor(states)

        # Calculate action probabilities of the student by applying softmax
        Q_a_student = self.student_network(states)
        logits_student = F.softmax(Q_a_student.detach())
        criterion = nn.CrossEntropyLoss()
        teacher_loss = criterion(logits_teacher, logits_student)


        self.teacher_optimizer.zero_grad()
        teacher_loss.backward()
        clip_grad_norm_(self.teacher_actor.parameters(), 1.)
        self.teacher_optimizer.step()

        return teacher_loss
        
    # Code for evaluating OOD data samples 
    def evaluate(self, experiences):
        state, action = experiences
        with torch.no_grad():
            current_Qs = self.student_network(state).detach()
            action_ex, prob = self.get_action_teacher(state).detach()
            selected_actions = torch.argmax(prob, dim=1).to(self.device)
            # print(f'action ------- {selected_actions} ------- {action}')
            selected_actions = selected_actions.view(action.shape)
            # calculate regularizing loss
            max_Q_values, _ = torch.max(current_Qs, dim=1)
            R_loss = torch.mean((max_Q_values- current_Qs.gather(1, selected_actions).squeeze(1)))
        return R_loss.detach().cpu().item()



    def cql_loss(self, q_values, current_action):
        """Computes the CQL loss for a batch of Q-values and actions."""
        logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
        q_a = q_values.gather(1, current_action)
    
        return (logsumexp - q_a).mean()

    def learn(self, experiences, ep):
        
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_a_s = self.student_network(states)
        # print(f'Q_a_s ======= {Q_a_s}')
        Q_expected = Q_a_s.gather(1, actions)
        
        cql1_loss = self.cql_loss(Q_a_s, actions)

        bellman_error = F.mse_loss(Q_expected, Q_targets)

        # If use teacher is true then integrate the rule loss otherwise set it to 0
        if self.use_teacher:
            rule_loss = self.calculate_teacher_loss(states, Q_expected, ep)
        else:
            rule_loss = F.mse_loss(Q_expected, Q_expected)
        
        # Add a rule regulizer term
        self.eps_rule = self.eps_rule-0.1
        if self.eps_rule<0:
            self.eps_rule = 0
        q1_loss = cql1_loss + 0.5 * bellman_error + self.lam * rule_loss # 0.5 works best for lunar lander 0.9 for cartpole
        
        self.optimizer.zero_grad()
        q1_loss.backward()
        clip_grad_norm_(self.student_network.parameters(), 1.)
        self.optimizer.step()

        # ------------------- update target student_network ------------------- #
        self.soft_update(self.student_network, self.target_net)
        return q1_loss.detach().item(), cql1_loss.detach().item(), bellman_error.detach().item(), rule_loss.detach().item()
        
        
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


"===========================CQL Agent using Huber Loss================================="
class CQL():

    def __init__(self,
                 state_size,
                 action_size,
                 device=None,
                 config = None):
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

        # parameters for configuration file
        self.env_name = config.env
        self.lam = config.lam
        self.lr = config.lr
        self.use_teacher = config.use_teach
        self.warm_start = config.warm_start
        self.teacher_update = config.teacher_update
        self.tlr =  config.tlr

        # loss function
        self.huber = nn.SmoothL1Loss()
        self.ce = nn.CrossEntropyLoss()

        # Number of training iterations
        self.iterations = 0

        # After how many training steps 'snap' target to main network?
        self.target_update_freq = 100

        # Q-Networks
        self.student_network = Critic(state_size[0], action_size, 42).to(self.device)

        self.student_network_target = copy.deepcopy(self.student_network)

        # Optimization
        self.optimizer = torch.optim.Adam(params=self.student_network.parameters(), lr=self.lr)

        # Teacher initialization for cartpole env
        if 'Cart' in self.env_name:
            self.teacher_actor = init_cart_nets('one_hot')
        # Teacher initialization for lunar lander env
        if 'Lunar' in self.env_name:
            self.teacher_actor = init_lander_nets('one_hot')
        if 'Mountain' in self.env_name:
            self.teacher_actor = init_mountain_nets('one_hot')
        if 'Lava' in self.env_name:
            self.teacher_actor = init_minigrid_nets('one_hot')
        if 'Dynamic' in self.env_name:
            self.teacher_actor = init_dynamic_nets('one_hot')

            # Initialize optimizer for teacher critic will use this for teacher training later
        if self.use_teacher:
            self.teacher_optimizer = optim.Adam(self.teacher_actor.parameters(), lr=self.tlr)

        # temperature parameter
        self.alpha = 0.1

    def get_action(self, state, epsilon=0):
        self.student_network.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            actions = self.student_network(state).cpu()
            action = np.argmax(actions.data.numpy())
        self.student_network.train()
        return [action]

    def get_deter_action(self, state):
        # state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.student_network.eval()
        with torch.no_grad():
            action_values = self.student_network(state.unsqueeze(0))
            # print(f'state ----- {state} ----action_values--- {action_values}')
        self.student_network.train()
        action = np.argmax(action_values.cpu().data.numpy(), axis=1)
        return action

    # Get action from the teacher network
    # The network return a probability distribution over the action
    def get_action_teacher(self, observation):
        obs = torch.Tensor(observation)
        self.teacher_actor.eval()
        probs = self.teacher_actor(obs)
        action = np.argmax(probs.cpu().data.numpy())
        self.teacher_actor.train()
        return action, probs.cpu() 
    
    #Function to check the entropy of the actions predicted
    def calculate_entropy(self, predictions):
        # Convert predictions to probabilities using softmax
        probs = torch.nn.functional.softmax(predictions, dim=-1)
        # Calculate entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        return entropy.mean()

    #Function to calculate the uncertainity of the states over 100 forward passes
    def calculate_uncertainity(self,action_rule,action_pred,states_mismatch):
        num_forward_passes = 10
        # self.student_network.eval()
        # Lists to store predictions
        predictions_pred = []
        predictions_rule = []
        for i in range(num_forward_passes):
            Q_val = self.student_network(states_mismatch, training=False).detach()
            # Gather the Q values for rule action and student action
            Q_val_pred = Q_val.gather(1, action_pred)
            Q_val_rule = Q_val.gather(1, action_rule)
            # Append predictions to lists
            predictions_pred.append(Q_val_pred.cpu().numpy())
            predictions_rule.append(Q_val_rule.cpu().numpy())
        # Convert predictions to tensors
        predictions_pred = torch.tensor(predictions_pred)
        predictions_rule = torch.tensor(predictions_rule)
        # Calculate the variance for predictions over 10 forward passes
        variance_pred = torch.var(predictions_pred, dim=0)
        variance_rule = torch.var(predictions_rule, dim=0)
        # self.student_network.train()
        return torch.mean(variance_pred), torch.mean(variance_rule)
    
    # This is main function for integration of teacher loss with the student network
    # Reduce the Q value for the stundent action
    # Increase the Q value for the teacher predicted action
    def calculate_teacher_loss(self, states, Q_expected, ep, steps):
        action_rule = []
        action_pred = []
        states_mismatch = []
        rule_error = 0
        # print(f'Q_expected ======= {Q_expected}')
        state_copy = copy.deepcopy(states)
        for s in state_copy:
            condition = check_condition(s,self.env_name)
            # Triggering in conditions for which the rules are written for, specific to environment
            if (condition):
                # Get the predicted action from the oriinal student_network
                predicted_action = self.get_deter_action(s)

                # selct actions using the rule based DDT
                rule_action, prob = self.get_action_teacher(s)

                # Increase Q value for rule action and decrease Q value for predicted action if there is an action mismatch
                # If actions dont match for a particular state add it to the list of states
                if(predicted_action!=rule_action):
                    # print(f'state mismatch ======= {s.cpu().data.numpy()} ========== {rule_action} ========= {predicted_action}')
                    states_mismatch.append(s.cpu().data.numpy())             
                    action_rule.append(rule_action)
                    action_pred.append(predicted_action[0])  

        if len(states_mismatch)!=0:
            action_rule = torch.from_numpy(np.vstack([a for a in action_rule])).long().to(self.device)
            action_pred = torch.from_numpy(np.vstack([a for a in action_pred])).long().to(self.device)
            states_mismatch = torch.from_numpy(np.vstack([s for s in states_mismatch])).float().to(self.device)
            # print(f'action_pred ======= {action_pred}')

            # Calculate Q values for only the states with mismatched action
            Q_val = self.student_network(states_mismatch)
            
            # Gathe the Q values for rule action and student action
            Q_val_pred = Q_val.gather(1,action_pred)
            Q_val_rule = Q_val.gather(1,action_rule)
            
            # Shift the Q value of student action to teacher action
            if ep<self.warm_start:
                rule_error = F.mse_loss(Q_val_pred, Q_val_rule) + F.mse_loss(Q_val_rule, Q_val_pred)
            else:
                # Do not update the student network in this case and train the teacher network
                # Check for high Q value and low uncertaininty
                if torch.mean(Q_val_rule) < torch.mean(Q_val_pred):
                    if 'Mountain' in self.env_name:
                        rule_error = F.mse_loss(Q_val_pred, Q_val_rule) + F.mse_loss(Q_val_rule, Q_val_pred)
                    else:
                        # set rule error to be 0
                        rule_error = F.mse_loss(Q_expected, Q_expected)
                        
                    # To not do uncertainity check every epoch and to reduce the update on the teacher network    
                    if ep%self.teacher_update ==0 and steps<2:
                        var_student, var_teacher = self.calculate_uncertainity(action_rule,action_pred,states_mismatch)
                        if var_student < var_teacher:
                            teacher_loss = self.teacher_learn(states_mismatch)
                            print(f'--------Training teacher-------------{teacher_loss}----------{var_student}------- {var_teacher}')
                            rule_error = F.mse_loss(Q_expected, Q_expected)
                        else:
                            rule_error = F.mse_loss(Q_val_pred, Q_val_rule) + F.mse_loss(Q_val_rule, Q_val_pred)
                else:
                    # If Q value not higher apply rule loss 
                    rule_error = F.mse_loss(Q_val_pred, Q_val_rule) + F.mse_loss(Q_val_rule, Q_val_pred)
        else:
            # else if no state match domain condition set the rule error to be 0
            rule_error = F.mse_loss(Q_expected, Q_expected)

        # print(rule_error)
        
        return rule_error

    # Code for updating the teacher network based on uncertainity
    # Uses cross entropy loss between the student logits and the teacher logits
    def teacher_learn(self,states):
        states = states
        logits_teacher = self.teacher_actor(states)

        # Calculate action probabilities of the student by applying softmax
        Q_a_student = self.student_network(states)
        with torch.no_grad():
            logits_student = F.softmax(Q_a_student.detach())
        criterion = nn.CrossEntropyLoss()
        teacher_loss = criterion(logits_teacher, logits_student)

        self.teacher_optimizer.zero_grad()
        teacher_loss.backward()
        clip_grad_norm_(self.teacher_actor.parameters(), 1.)
        self.teacher_optimizer.step()

        return teacher_loss
    
    def policy(self, state, eval=False):

        # set networks to eval mode
        self.student_network.eval()

        if eval:
            eps = self.eval_eps
        else:
            eps = max(self.slope * self.iterations + self.initial_eps, self.end_eps)

        # epsilon greedy policy
        if self.rng.uniform(0, 1) > eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_val = self.student_network(state).cpu()
                return q_val.argmax().item(), q_val, np.nan
        else:
            return self.rng.integers(self.action_space), np.nan, np.nan

    # Code for evaluating OOD data samples 
    def evaluate(self, experiences):
        state, action = experiences
        with torch.no_grad():
            current_Qs = self.student_network(state).detach()
            prob = self.teacher_actor(state).detach()
            selected_actions = torch.argmax(prob, dim=1).to(self.device)
            # print(f'action ------- {selected_actions} ------- {action}')
            selected_actions = selected_actions.view(action.shape)
            # calculate regularizing loss
            max_Q_values, _ = torch.max(current_Qs, dim=1)
            O_loss = torch.mean((max_Q_values- current_Qs.gather(1, selected_actions).squeeze(1))).detach().cpu().item()
        return O_loss

    def learn(self, experiences, ep, steps):
        # Sample replay buffer
        state, action, reward, next_state, not_done = experiences

        # set networks to train mode
        self.student_network.train()
        self.student_network_target.train()

        ### Train main network

        # Compute the target Q value
        with torch.no_grad():
            q_val = self.student_network(next_state)
            next_action = q_val.argmax(dim=1, keepdim=True)
            target_Q = reward + not_done * self.discount * self.student_network_target(next_state).gather(1, next_action)

        # Get current Q estimate
        current_Qs = self.student_network(state)
        current_Q = current_Qs.gather(1, action)

        # Compute Q loss (Huber loss)
        Q_loss = self.huber(current_Q, target_Q)

        # calculate regularizing loss
        R_loss = torch.mean(self.alpha * (torch.logsumexp(current_Qs, dim=1) - current_Qs.gather(1, action).squeeze(1)))

        if self.use_teacher:
            rule_loss = self.calculate_teacher_loss(state, current_Qs, ep, steps)
        else:
            # Set rule loss 0 when not using teacher
            rule_loss = F.mse_loss(current_Qs, current_Qs)
        loss_val = rule_loss.detach().item()
        # Optimize the Q
        self.optimizer.zero_grad()
        '''if loss_val!=0:
            (rule_loss).backward()
        else:
            (Q_loss + R_loss).backward()'''
        loss = Q_loss + R_loss + self.lam * rule_loss
        loss.backward()
        self.optimizer.step()

        self.iterations += 1
        # Update target network by full copy every X iterations.
        if self.iterations % self.target_update_freq == 0:
            self.student_network_target.load_state_dict(self.student_network.state_dict())

        return loss.detach().cpu().item(), R_loss.detach().cpu().item(), Q_loss.detach().cpu().item(), rule_loss.detach().item()

    def get_name(self) -> str:
        return "ConservativeQLearning"