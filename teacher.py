import numpy as np
from prolonet import ProLoNet
import torch
from networks import ANetwork, MActor, Actor

# This is the initialization for cartpole environment based DDT
# This DDT is based on cartpole heuristics given in rules.py

def init_cart_nets(distribution, use_gpu=False, vectorized=False, randomized=False):
    if torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False

    dim_in = 4
    dim_out = 2
    w1 = np.zeros(dim_in)
    c1 = np.ones(dim_in)*2
    w1[0] = 1  # cart position
    c1[0] = -1  # > -1

    w2 = np.zeros(dim_in)
    c2 = np.ones(dim_in)*2
    w2[0] = -1  # negative position
    c2[0] = -1  # < 1  (so if positive < 4)

    w3 = np.zeros(dim_in)
    c3 = np.ones(dim_in)*2
    w3[2] = -1  # pole angle
    c3[2] = 0  # < 0

    w4 = np.zeros(dim_in)
    c4 = np.ones(dim_in)*2
    w4[2] = -1
    c4[2] = 0  # < 0

    w5 = np.zeros(dim_in)
    c5 = np.ones(dim_in)*2
    w5[2] = -1
    c5[2] = 0  # < 0

    w6 = np.zeros(dim_in)
    c6 = np.ones(dim_in)*2
    w6[1] = -1  # cart velocity
    c6[1] = 0  # < 0

    w7 = np.zeros(dim_in)
    c7 = np.ones(dim_in)*2
    w7[1] = 1  # cart velocity
    c7[1] = 0  # > 0

    w8 = np.zeros(dim_in)
    c8 = np.ones(dim_in)*2
    w8[3] = 1  # pole rate
    c8[3] = 0  # > 0

    w9 = np.zeros(dim_in)
    c9 = np.ones(dim_in)*2
    w9[2] = -1
    c9[2] = 0

    w10 = np.zeros(dim_in)
    c10 = np.ones(dim_in)*2
    w10[3] = -1
    c10[3] = 0

    w11 = np.zeros(dim_in)
    c11 = np.ones(dim_in)*2
    w11[2] = -1
    c11[2] = 0

    init_weights = [
        w1,
        w2,
        w3,
        w4,
        w5,
        w6,
        w7,
        w8,
        w9,
        w10,
        w11,
    ]
    init_comparators = [
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        c8,
        c9,
        c10,
        c11,
    ]

    if distribution == 'one_hot':
        leaf_base_init_val = 0.
        leaf_target_init_val = 1.
    elif distribution == 'soft_hot':
        leaf_base_init_val = 0.1 / dim_out
        leaf_target_init_val = 0.9
    else:  # uniform
        leaf_base_init_val = 1.0 / dim_out
        leaf_target_init_val = 1.0 / dim_out
    leaf_base = [leaf_base_init_val] * 2

    l1 = [[], [0, 2], leaf_base.copy()]
    l1[-1][1] = leaf_target_init_val  # Right

    l2 = [[0, 1, 3], [], leaf_base.copy()]
    l2[-1][0] = leaf_target_init_val  # Left

    l3 = [[0, 1], [3], leaf_base.copy()]
    l3[-1][1] = leaf_target_init_val  # Right

    l4 = [[0, 4], [1], leaf_base.copy()]
    l4[-1][0] = leaf_target_init_val  # Left

    l5 = [[2, 5, 7], [0], leaf_base.copy()]
    l5[-1][1] = leaf_target_init_val  # Right

    l6 = [[2, 5], [0, 7], leaf_base.copy()]
    l6[-1][0] = leaf_target_init_val  # Left

    l7 = [[2, 8], [0, 5], leaf_base.copy()]
    l7[-1][0] = leaf_target_init_val  # Left

    l8 = [[2], [0, 5, 8], leaf_base.copy()]
    l8[-1][1] = leaf_target_init_val  # Right

    l9 = [[0, 6, 9], [1, 4], leaf_base.copy()]
    l9[-1][0] = leaf_target_init_val  # Left

    l10 = [[0, 6], [1, 4, 9], leaf_base.copy()]
    l10[-1][1] = leaf_target_init_val  # Right

    l11 = [[0, 10], [1, 4, 6], leaf_base.copy()]
    l11[-1][0] = leaf_target_init_val  # Left

    l12 = [[0], [1, 4, 6, 10], leaf_base.copy()]
    l12[-1][1] = leaf_target_init_val  # Right

    init_leaves = [
        l1,
        l2,
        l3,
        l4,
        l5,
        l6,
        l7,
        l8,
        l9,
        l10,
        l11,
        l12,
    ]
    if not vectorized:
        init_comparators = [comp[comp != 2] for comp in init_comparators]
    if randomized:
        init_weights = [np.random.normal(0, 0.1, dim_in) for w in init_weights]
        init_comparators = [np.random.normal(0, 0.1, c.shape) for c in init_comparators]
        init_leaves = [[l[0], l[1], np.random.normal(0, 0.1, dim_out)] for l in init_leaves]
        init_weights = None
        init_comparators = None
        init_leaves = 4
    action_network = ProLoNet(input_dim=dim_in,
                              output_dim=dim_out,
                              weights=init_weights,
                              comparators=init_comparators,
                              leaves=init_leaves,
                              alpha=1,
                              is_value=False,
                              device='cuda' if use_gpu else 'cpu',
                              vectorized=vectorized)
    
    anetwork_local = Actor(4, 2, 0)
    anetwork_local.load_state_dict(torch.load('model/cartpoleteacher.pt'))
    action_network = anetwork_local
    # This part might not be important for us
    value_network = ProLoNet(input_dim=dim_in,
                             output_dim=dim_out,
                             weights=init_weights,
                             comparators=init_comparators,
                             leaves=init_leaves,
                             alpha=1,
                             is_value=True,
                             device='cuda' if use_gpu else 'cpu',
                             vectorized=vectorized)
    if use_gpu:
        action_network = action_network.cuda()
        value_network = value_network.cuda()
    return action_network

# Check the conditions for which this neural network is certain
def check_condition(s,env_name):
    if 'Cart' in env_name:
        '''if s[3]>1:
            if s[3]>-0:
                if s[2]>-1:
                    return True
        else:
            if s[2]>0.1:         
                return True
            else:
                return True'''
        if(s[3]>0.44):
            return True
        else:
            return False
    if 'Lunar' in env_name:
        if(s[5]<0.04):
            return True
    if 'Mountain' in env_name:
        if s[0] < -0.5:
            if s[1] < -0.01:
                return True
            else:
                return True
        else:
            if s[1] > 0.01:
                return True
    # Condition for minigrid environment
    if 'Lava' in env_name:
        # Since the rule says move forward if no wall or lava this should be true always
        return True
    if 'Dynamic' in env_name:
        # Since the rule says move forward if no wall or lava this should be true always
        return True
    
    return False


# This DDT is based on lunar lander heuristics given in rules.py
def init_lander_nets(distribution, use_gpu=False, vectorized=False, randomized=False):
    if torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False
    dim_in = 8
    dim_out = 4
    w1 = np.zeros(dim_in)
    c1 = np.ones(dim_in) * 2
    w1[5] = -1  
    c1[5] = 0.5  

    w2 = np.zeros(dim_in)
    c2 = np.ones(dim_in) * 2
    w2[3] = 1  
    c2[3] = -0.3  

    w3 = np.zeros(dim_in)
    c3 = np.ones(dim_in) * 2
    w3[5] = 1  
    c3[5] = -0.22  

    w4 = np.zeros(dim_in)
    c4 = np.ones(dim_in) * 2
    w4[4] = 1  
    c4[4] = -0.04  

    w5 = np.zeros(dim_in)
    c5 = np.ones(dim_in) * 2
    w5[6] = 1  # binary flag indicating if the tip of the pole is below the threshold
    c5[6] = 1  # == 1

    init_weights = [w1, w2, w3, w4, w5]
    init_comparators = [c1, c2, c3, c4, c5]


    if distribution == 'one_hot':
        leaf_base_init_val = 0.
        leaf_target_init_val = 1.
    elif distribution == 'soft_hot':
        leaf_base_init_val = 0.1 / dim_out
        leaf_target_init_val = 0.9
    else:  # uniform
        leaf_base_init_val = 1.0 / dim_out
        leaf_target_init_val = 1.0 / dim_out
    leaf_base = [leaf_base_init_val] * dim_out

    # Decision Tree Nodes
    l1 = [[], [0], leaf_base.copy()]  # obs[5] > 0.04: action = 3
    l1[-1][3] = leaf_target_init_val  # Action 3

    l2 = [[1], [2], leaf_base.copy()]  # obs[3] <= -0.35
    l2[-1][0] = leaf_target_init_val  # Action 0

    l3 = [[], [3], leaf_base.copy()]  # obs[3] > -0.35, obs[5] <= -0.22
    l3[-1][0] = leaf_target_init_val  # Action 0

    l4 = [[4], [], leaf_base.copy()]  # obs[3] > -0.35, obs[5] <= -0.22, obs[4] <= -0.04
    l4[-1][1] = leaf_target_init_val  # Action 1

    l5 = [[], [], leaf_base.copy()]  # Otherwise
    l5[-1][0] = leaf_target_init_val  # Action 0

    init_leaves = [
        l1,
        l2,
        l3,
        l4,
        l5
    ]
    if not vectorized:
        init_comparators = [comp[comp != 2] for comp in init_comparators]
    if randomized:
        init_weights = [np.random.normal(0, 0.1, dim_in) for w in init_weights]
        init_comparators = [np.random.normal(0, 0.1, c.shape) for c in init_comparators]
        init_leaves = [[l[0], l[1], np.random.normal(0, 0.1, dim_out)] for l in init_leaves]
        init_weights = None  # None for random init within network
        init_comparators = None
        init_leaves = 16  # This is one more than intelligent, but best way to get close with selector
    action_network = ProLoNet(input_dim=dim_in,
                              output_dim=dim_out,
                              weights=init_weights,
                              comparators=init_comparators,
                              leaves=init_leaves,
                              alpha=1.,
                              is_value=False,
                              vectorized=vectorized,
                              device='cuda' if use_gpu else 'cpu')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    anetwork_local = ANetwork(8, 4, 0).to(device)
    anetwork_local.load_state_dict(torch.load('model/lunarteachteacher2.pth'))
    action_network = anetwork_local
    value_network = ProLoNet(input_dim=dim_in,
                             output_dim=dim_out,
                             weights=init_weights,
                             comparators=init_comparators,
                             leaves=init_leaves,
                             alpha=1.,
                             is_value=True,
                             vectorized=vectorized,
                             device='cuda' if use_gpu else 'cpu')
    return action_network

# Differentiable descision tree for mountain car
def init_mountain_nets(distribution, use_gpu=False, vectorized=False, randomized=False):
    if torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False
    dim_in = 2
    dim_out = 3
    w1 = np.zeros(dim_in)
    c1 = np.ones(dim_in) * 2
    w1[0] = -1  
    c1[0] = -0.5 # position < -0.5 

    w2 = np.zeros(dim_in)
    c2 = np.ones(dim_in) * 2
    w2[1] = -1  
    c2[1] = -0.01 # velocity < -0.01
    
    w3 = np.zeros(dim_in)
    c3 = np.ones(dim_in) * 2
    w3[1] = 1  
    c3[1] = 0.01 # velocity > 0.01

    init_weights = [w1, w2, w3]
    init_comparators = [c1, c2, c3]


    if distribution == 'one_hot':
        leaf_base_init_val = 0.
        leaf_target_init_val = 1.
    elif distribution == 'soft_hot':
        leaf_base_init_val = 0.1 / dim_out
        leaf_target_init_val = 0.9
    else:  # uniform
        leaf_base_init_val = 1.0 / dim_out
        leaf_target_init_val = 1.0 / dim_out
    leaf_base = [leaf_base_init_val] * dim_out

    # Decision Tree Nodes
    l1 = [[0,1], [], leaf_base.copy()]  # position < -0.5
    l1[-1][0] = leaf_target_init_val  # Action 0

    l2 = [[0], [1], leaf_base.copy()]  # position >= -0.5, velocity > 0.01
    l2[-1][2] = leaf_target_init_val  # Action 2

    l3 = [[2], [0], leaf_base.copy()]  # position >= -0.5, velocity <= 0.01
    l3[-1][2] = leaf_target_init_val  # Action 0

    l4 = [[], [0,2], leaf_base.copy()]  # position >= -0.5, velocity > 0.01
    l4[-1][0] = leaf_target_init_val  # Action 0

    init_leaves = [
        l1,
        l2,
        l3,
        l4
    ]
    if not vectorized:
        init_comparators = [comp[comp != 2] for comp in init_comparators]
    if randomized:
        init_weights = [np.random.normal(0, 0.1, dim_in) for w in init_weights]
        init_comparators = [np.random.normal(0, 0.1, c.shape) for c in init_comparators]
        init_leaves = [[l[0], l[1], np.random.normal(0, 0.1, dim_out)] for l in init_leaves]
        init_weights = None  # None for random init within network
        init_comparators = None
        init_leaves = 16  # This is one more than intelligent, but best way to get close with selector
    action_network = ProLoNet(input_dim=dim_in,
                              output_dim=dim_out,
                              weights=init_weights,
                              comparators=init_comparators,
                              leaves=init_leaves,
                              alpha=1.,
                              is_value=False,
                              vectorized=vectorized,
                              device='cuda' if use_gpu else 'cpu')
    #anetwork_local = MActor(2, 3, hidden_dims=(128,128))
    #anetwork_local.load_state_dict(torch.load('data/mountaincar.pth'))
    anetwork_local = Actor(2, 3, 0)
    anetwork_local.load_state_dict(torch.load('model/mountaincarteacher.pt'))
    action_network = anetwork_local
    value_network = ProLoNet(input_dim=dim_in,
                             output_dim=dim_out,
                             weights=init_weights,
                             comparators=init_comparators,
                             leaves=init_leaves,
                             alpha=1.,
                             is_value=True,
                             vectorized=vectorized,
                             device='cuda' if use_gpu else 'cpu')
    return action_network

# Teacher network for minigrid lavagaps
def init_minigrid_nets(use_gpu=False):
    anetwork_local = Actor(98, 3, 0)
    anetwork_local.load_state_dict(torch.load('model/minigriteacher.pt'))
    action_network = anetwork_local
    return action_network

def init_dynamic_nets(use_gpu=False):
    anetwork_local = Actor(98, 3, 0)
    anetwork_local.load_state_dict(torch.load('model/dynamicran6teacher.pt'))
    action_network = anetwork_local
    return action_network