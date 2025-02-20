import numpy as np
import random
from networks import ANetwork
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
anetwork_local = ANetwork(8, 4, 0).to(device)
anetwork_local.load_state_dict(torch.load('model/lunarteacher.pth'))

def get_action_heur_cartpole1(obs):
    action = None
    if obs[3]>1:
        if obs[3]>-0:
            if obs[2]>-1:
                action = 1
    else:
        if obs[2]>0:
            
            action = 1
        else:
            action = 0
    return [action]



# Action heuristics for cartpole environment 
def get_action_heur_cartpole(observation):
        action = None
        if observation[3] > -0.1:
            if observation[3] > -0.1:
                if(observation[0]>1):
                    if observation[0] > -1:
                        if observation[0] < 1:
                            if observation[2] < 0:
                                action = 0
                            else:
                                action = 1
                        else:
                            if observation[2] < 0:
                                action = 0
                            else:
                                if observation[1] > 0:
                                    if observation[3] < 0:
                                        action = 0
                                    else:
                                        action = 1
                                else:
                                    if observation[2] < 0:
                                        action = 0
                                    else:
                                        action = 1
                    else:
                        if observation[2] < 0:
                            if observation[1] < 0:
                                if observation[3] > 0:
                                    action = 1
                                else:
                                    action = 0
                            else:
                                if observation[2] < 0:
                                    action = 0
                                else:
                                    action = 1
        return [action]

def get_action_heur_mountaincar(observation):
    action = None
    position = observation[0]
    velocity = observation[1]
    if position > -0.8:
        if position < -0.5:
            if velocity < -0.01:
                action = 0
            else:
                action = 2
        else:
            if velocity > 0.01:
                action = 2
    return [action]

# Action heuristics for lunar lander environment 
def get_action_heur_lunar(obs):
    action = None
    if obs[5]<0.04:
        if obs[3]> -0.35:
            action = get_action_heur_lunar_q(obs)
        else:
            if obs[5]>-0.22:
                action = get_action_heur_lunar_q(obs)
            else:
                if obs[4]>-0.04:
                    action = get_action_heur_lunar_q(obs)
    
    return [action]

def get_action_heur_lunar_q(obs):
    obs = torch.from_numpy(obs).float().to(device)
    anetwork_local.eval()
    with torch.no_grad():
        action_values = anetwork_local(obs)
    action = np.argmax(action_values.cpu().data.numpy())
    return action

def get_action_cartpole_heur2(obs):
    action = None
    if obs[3]>0.44:
        if obs[3]>-0.3:
            if obs[2]>-0.41:
                action = 1
            else:
                action = 0
        else:
            if obs[2]>0.0:
                action = 0
            else:
                action = 1
    return [action]

# Perfect rules for the cartpole environment
# position velocity angle angular velocity
# Rules for cartpole environment
def get_action_cartpole(obs):
    action = random.choice([0, 1])
    if obs[3]>0.44:
        if obs[3]>-0.3:
            if obs[2]>-0.41:
                action = 1
            else:
                action = 0
        else:
            if obs[2]>0.0:
                action = 0
            else:
                action = 1
    else:
        if obs[2]>0.01:
            
            action = 1
        else:
            action = 0
    return action

# Rule with minigrid lavagaps environment 
def get_action_heur_lava(obs):
    action = None
    if obs[68] in [0.2, 0.9]:
        if obs[52] == 0.9:
            # Wall or lava not in left move left
            if obs[40] not in [0.2, 0.9]:
                action = 0
            # Wall or lava not in right move right
            elif obs[68] not in [0.2, 0.9]:
                action = 1
            else:
                action = random.choice([0, 1])
    return [action]

def get_action_heur_dynamic(obs):
    action = None
    if obs[68] in [0.2, 0.9]:
        if obs[52] == 0.6:
            # Wall or lava not in left move left
            if obs[40] not in [0.2, 0.6]:
                action = 0
            # Wall or lava not in right move right
            elif obs[68] not in [0.2, 0.6]:
                action = 1
            else:
                action = random.choice([0, 1])
    return [action]

# Calculate the soft deviation from the rule
def calculate_soft_deviation(state):
    # This is only for the rule pole velocity > -0.3 and pole angle > -0.41 -> right [action = 1]
    pred_1 = 1
    pred_2 = 0
    if state[3]>-0.3:
        pred_1 = 1
        if state[2]>-0.41:
            pred_2 = 1
        else:
            pred_2 = 0
    else:
        pred_1 = 0
    # pred_1 = state[3] - (-0.3)
    # pred_2 = state[2] - (-0.41)
    state = np.append(state,pred_1)
    state = np.append(state,pred_2)
    return state