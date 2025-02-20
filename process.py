'''
File process the data into buffer structure 
["state", "action", "reward", "next_state", "done"]
'''


import pickle
from collections import deque, namedtuple
import pandas as pd
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Your script description here.')

# Add an argument for exp_name with a default value
parser.add_argument('--exp_name', type=str, default='data/ex7/LunarLander-v2_run3_noisy.pkl', help='Specify the experiment name')

# Parse the command line arguments
args = parser.parse_args()

# Access the value of exp_name
exp_name = args.exp_name

new_name = exp_name.split(".")[0]

with open(exp_name, "rb") as f:
        experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        buffer = pickle.load(f)
memory = deque(maxlen=len(buffer.state))  
for i in range(len(buffer.state)-1):
        e = experience(buffer.state[i], buffer.action[i], buffer.reward[i], buffer.state[i+1], buffer.not_done[i])
        memory.append(e)

li_mem = list(memory)
print(len(li_mem))
df = pd.DataFrame(li_mem)
df.to_pickle(new_name+'conv.pkl')

print(len(buffer.state))
print(memory[0])