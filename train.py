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
import glob
from utils import save, collect_random, process_data
import random
from agent import CQLAgent, CQL
from rules import *
import pandas as pd
import ast
import statistics
import copy
from wrappers import FlatImgObsWrapper, RestrictMiniGridActionWrapper

def read_config_file(config_file_path):
    config = configparser.ConfigParser()
    config.read(config_file_path)
    return config

'''
Function to read the configuration for each environemnt from a config file
'''

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
    data_len = len(df_filtered.index)
    data_used = round(data_len * config.data_percent)
    buffer_ood = ReplayBuffer(buffer_size=config.buffer_size, batch_size=32, device=device)

    # Use the OOD data for training
    df_head = df_filtered.head(data_used)

    # To check what is happening in the case of OOD Data
    df_sampled = df.sample(frac = config.data_percent)
    df = df_head

    # Find the intersection of actual buffere and sampled buffer
    if config.use_ood:
        df_new = df_head.copy(deep=True)
        df_new['state'] = df_new['state'].apply(tuple)
        df_sampled['state'] = df_sampled['state'].apply(tuple)
        intersection_df = pd.merge(df_new, df_sampled, on=['state'], how='inner')
        intersection_df['state'] = intersection_df['state'].apply(np.array)

        # Make a new ood buffer
        for ind in intersection_df.index:
            buffer_ood.add_ood(intersection_df['state'][ind],intersection_df.iloc[:, 1][ind])
        print(f'------{len(buffer_ood.memory_ood)}-------{buffer_ood.memory_ood[0]}')
    
    # Add data processed from CSV to buffer
    for ind in df.index:
        buffer.add(df['state'][ind],df['action'][ind],df['reward'][ind],df['next_state'][ind],df['done'][ind])
    print(f'------{len(buffer.memory)}-------{buffer.memory[0]}')
    return buffer, buffer_ood

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
    
    eps = 1.
    d_eps = 1 - config.min_eps
    steps = 0
    average10 = deque(maxlen=10)
    total_steps = 0

    # This code is used for training the student_network online without wanddb logs and saving data to buffer
    if not config.use_wandb:

        
        agent = CQLAgent(state_size=env.observation_space.shape,
                            action_size=env.action_space.n,
                            device=device, config = config)

        buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=32, device=device)
        
        collect_random(env=env, dataset=buffer, num_samples=10000)
        
        if config.log_video:
            env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x%10==0, force=True)

        for i in range(1, config.episodes+1):
            state = env.reset()
            episode_steps = 0
            rewards = 0
            while True:
                action = agent.get_action(state, epsilon=eps)
                steps += 1
                next_state, reward, done, _ = env.step(action[0])
                buffer.add(state, action, reward, next_state, done)
                loss, cql_loss, bellmann_error, rule_loss = agent.learn(buffer.sample(64))
                state = next_state
                rewards += reward
                episode_steps += 1
                eps = max(1 - ((steps*d_eps)/config.eps_frames), config.min_eps)
                if done:
                    break

            average10.append(rewards)
            total_steps += episode_steps
            print("Episode: {} | Reward: {} | Q Loss: {} | Steps: {}".format(i, rewards, loss, steps,))

        li_mem = list(buffer.memory)
        print(len(li_mem))
        df = pd.DataFrame(li_mem)
        df.to_pickle('data/'+config.env+'.pkl')

    # This is the main offline training part
    else:
    
        with wandb.init(project="CQL", name=config.run_name, config=config):
            
            agent = CQL(state_size=env.observation_space.shape,
                                action_size=env.action_space.n,
                                device=device, config = config)

            #wandb.watch(agent.student_network, log="gradients", log_freq=10)

            buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=32, device=device)
            
            # collect_random(env=env, dataset=buffer, num_samples=10000)
            
            if config.log_video:
                env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x%10==0, force=True)

            # Load the buffer with offline data

            buffer, buffer_ood = read_transform_csv(config.data_file,buffer,config)

            for i in range(1, config.episodes+1):
                state = env.reset()
                episode_steps = 0
                rewards = 0

                # Evaluate the offline agent
                while True:
                    action = agent.get_action(state, epsilon=eps)

                    # Train the offline agent
                    # Train only till the env is solved
                    if config.use_teach:
                        if len(average10)==0 or np.mean(average10)<=config.threshold:
                            loss, cql_loss, bellmann_error, rule_loss = agent.learn(buffer.sample(32), i, episode_steps)
                        else:
                            loss, cql_loss, bellmann_error, rule_loss = 0, 0, 0, 0
                    else:
                        loss, cql_loss, bellmann_error, rule_loss = agent.learn(buffer.sample(32), i, episode_steps)

                    #Evaluate OOD buffer to see how the CQL exponential term is changing
                    cql_loss_buffer=0
                    if config.use_ood:
                        cql_loss_buffer = agent.evaluate(buffer_ood.sample_ood(256))
                    else:
                        cql_loss_buffer=0

                    steps += 1
                    next_state, reward, done, _ = env.step(action[0])

                    state = next_state
                    rewards += reward
                    episode_steps += 1
                    eps = max(1 - ((steps*d_eps)/config.eps_frames), config.min_eps)
                    if done:
                        break

                average10.append(rewards)
                            
                total_steps += episode_steps
                print("Episode: {} | Reward: {} | Q Loss: {} | Steps: {}".format(i, rewards, loss, steps,))
                
                wandb.log({"Reward": rewards,
                        "Average10": np.mean(average10),
                        "Steps": total_steps,
                        "Q Loss": loss,
                        "CQL Loss": cql_loss,
                        "Bellmann error": bellmann_error,
                        "Steps": steps,
                        "Epsilon": eps,
                        "Episode": i,
                        "Buffer size": buffer.__len__(),
                        "Rule Loss" : rule_loss,
                        "OOD Loss" : cql_loss_buffer})

                if (i %10 == 0) and config.log_video:
                    mp4list = glob.glob('video/*.mp4')
                    if len(mp4list) > 1:
                        mp4 = mp4list[-2]
                        wandb.log({"gameplays": wandb.Video(mp4, caption='episode: '+str(i-10), fps=4, format="gif"), "Episode": i})

                if i % config.save_every == 0:
                    save(config, save_name="CQL-DQN", model=agent.student_network, wandb=wandb, ep=0)

            print(f'standar deviation ------------- {statistics.stdev(average10)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process the configuration file.')
    
    # Define the command-line argument for the configuration file path
    parser.add_argument('--config_file', default='config/mountain.config', help='Path to the configuration file')

    # Parse the command-line arguments
    args = parser.parse_args()
    config_file = args.config_file
    config = get_config(config_file)
    train(config)

