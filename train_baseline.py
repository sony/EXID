'''
Created by : Anonymous for submision
This is the main file for implementing domain knowledge based OOD handling in oofline RL
'''

import gym
import numpy as np
from collections import deque
import torch
import wandb
import argparse, configparser
from buffer import ReplayBuffer
import random
from baselines import *
from rules import *
import pandas as pd
import statistics
import ast
from utils import save, collect_random, process_data
from wrappers import FlatImgObsWrapper, RestrictMiniGridActionWrapper

def read_config_file(config_file_path):
    config = configparser.ConfigParser()
    config.read(config_file_path)
    return config

def get_config(config_file):
    
    # Read from the configuration file given for each environment
    config = read_config_file(config_file)
    parser = argparse.ArgumentParser(description="DRL")

    # Add arguments dynamically based on the values in the config file
    for key, value in config.items('arguments'):
        # Extract data type and value
        type_str, raw_value = value.split(':', 1)
        data_type = eval(type_str)
        # Special handling for boolean values
        if data_type is bool:
            value = ast.literal_eval(raw_value)
        else:
            value = data_type(raw_value)
        print(f'key ------------- {key} ------------- value {value}')
        parser.add_argument(f'--{key}', type=type(value), default=value, help=f'Description for {key}')

    args, unknown = parser.parse_known_args()
    print(args.use_heur)

    return args

'''
Function to initialize buffer from offline data
file_path : path to offline data
buffer : buffer memory
config : config file for data percentage
'''
def read_transform_csv(file_path, buffer, config):
    """
    Convert pandas dataframe to a bffer object from oflline data
    """
    df = pd.read_pickle(file_path)
    df_filtered = process_data(df, config.data_type, config.env)
    data_len = len(df.index)
    data_used = round(data_len * config.data_percent)

    # n = 1150 # This parameter is specifically for minigrid dynamic obstacles
    # df = df_filtered.iloc[n:n+150] #df_filtered.head(data_used)
    # df = df_filtered.head(data_used)
    
    df = df_filtered.head(data_used)
    # print(df.head())
    # Add data processed from CSV to buffer
    for ind in df.index:
        buffer.add(df['state'][ind],df['action'][ind],df['reward'][ind],df['next_state'][ind],df['done'][ind])
    print(f'------{len(buffer.memory)}-------{buffer.memory[0]}')
    return buffer

'''
The main training loop
The agent is trained using buffer data and then continuously evaluated
'''
def train(config):

    # Set seed for reproducible result

    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    env = gym.make(config.env)
    if 'MiniGrid' in config.env:
        env = FlatImgObsWrapper(RestrictMiniGridActionWrapper(env))

    
    env.seed(config.seed)
    env.action_space.seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    steps = 0
    average10 = deque(maxlen=10)
    total_steps = 0

    
    with wandb.init(project="CQL", name=config.run_name, config=config):
        
        if config.algo_type == 'BC':
            agent = BehavioralCloning(state_size=env.observation_space.shape, 
                            action_size=env.action_space.n,
                            device=device)
        if config.algo_type == 'BCQ':
            agent = BCQ(state_size=env.observation_space.shape, 
                            action_size=env.action_space.n,
                            device=device)
            
        if config.algo_type == 'BVE':
            agent = BVE(state_size=env.observation_space.shape, 
                            action_size=env.action_space.n,
                            device=device)

        if config.algo_type == 'CQL':
            agent = CQL(state_size=env.observation_space.shape, 
                            action_size=env.action_space.n,
                            device=device)

        if config.algo_type == 'CRR':
            agent = CRR(state_size=env.observation_space.shape, 
                            action_size=env.action_space.n,
                            device=device)
            
        if config.algo_type == 'MCE':
            agent = MCE(state_size=env.observation_space.shape, 
                            action_size=env.action_space.n,
                            device=device)

        if config.algo_type == 'REM':
            agent = REM(state_size=env.observation_space.shape, 
                            action_size=env.action_space.n,
                            device=device)

        if config.algo_type == 'QRDQN':
            agent = QRDQN(state_size=env.observation_space.shape, 
                            action_size=env.action_space.n,
                            device=device)

        # wandb.watch(agent.actor, log="gradients", log_freq=10)

        buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=32, device=device)

        # Load the buffer with offline data

        buffer = read_transform_csv(config.data_file,buffer,config)

        for i in range(1, config.episodes+1):
            state = env.reset()
            episode_steps = 0
            rewards = 0

            # Evaluate the offline agent
            while True:

                if config.use_heur:
                    if 'Cart' in config.env:
                        action = get_action_heur_cartpole(state)
                    if 'Lunar' in config.env:
                        action = get_action_heur_lunar(state)
                    if 'Mountain' in config.env:
                        action = get_action_heur_mountaincar(state)
                    if action[0]==None:
                        action = agent.get_action(state)
                else:
                    action = agent.get_action(state)
                # Train the offline agent
                # Train only till the env is solved
                # print(f'average 10 ------------ {np.mean(average10)}')
                loss = agent.train(buffer.sample(32))


                steps += 1
                next_state, reward, done, _ = env.step(action[0])

                state = next_state
                rewards += reward
                episode_steps += 1

                if done:
                    break

            average10.append(rewards)
                        
            total_steps += episode_steps
            print("Episode: {} | Reward: {} | Q Loss: {} | Steps: {}".format(i, rewards, loss, steps,))
            
            wandb.log({"Reward": rewards,
                    "Average10": np.mean(average10),
                    "Steps": total_steps,
                    "Loss": loss,
                    "Steps": steps,
                    "Episode": i,
                    "Buffer size": buffer.__len__(),})
        print(f'standar deviation ------------- {statistics.stdev(average10)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process the configuration file.')
    
    # Define the command-line argument for the configuration file path
    parser.add_argument('--config_file', default='config/lunar.config', help='Path to the configuration file')

    # Parse the command-line arguments
    args = parser.parse_args()
    config_file = args.config_file
    config = get_config(config_file)
    train(config)
