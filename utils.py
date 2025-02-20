import torch
import numpy as np 
from rules import calculate_soft_deviation

def save(args, save_name, model, wandb, ep=None):
    import os
    save_dir = './trained_models/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
        wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
        wandb.save(save_dir + args.run_name + save_name + ".pth")

def collect_random(env, dataset, num_samples=200):
    state = env.reset()
    # state = calculate_soft_deviation(state)
    for _ in range(num_samples):
        # print(f' sample state ------- {state}')
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        # next_state = calculate_soft_deviation(next_state)
        dataset.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()
            # state = calculate_soft_deviation(state)

def entropy(values):
    probs = values.detach().cpu().numpy()
    # if entropy degrades
    if np.min(probs) < 1e-5:
        return 0
    return -np.sum(probs * np.log(probs))

# Create OOD data in order for CQL algorithm to fail
def process_data(df, buffer_type, env_name):
    if 'Mountain' in env_name:
        if buffer_type=='rep':
            filtered_df = df[df['state'].apply(lambda x: x[0] > -0.8 or x[0] > -0.2)]
            return filtered_df
        if buffer_type=='ns':
            filtered_df = df[df['state'].apply(lambda x: x[0] > -0.8 or x[0] > -0.2)]
            return filtered_df
    if 'Lunar' in env_name:
        if buffer_type=='rep':
            filtered_df = df[df['state'].apply(lambda x: x[4]>-0.04)]
            return filtered_df
        if buffer_type=='ns':
            filtered_df = df[df['state'].apply(lambda x: x[4]>-0.04)]
            return filtered_df
    if 'Cart' in env_name:
        if buffer_type=='er':
            filtered_df = df[df['state'].apply(lambda x: x[3] > -0.1)]
            return filtered_df
        if buffer_type=='rep':
            filtered_df = df[df['state'].apply(lambda x: x[3] > -0.2 or x[0] <-1)]
            return filtered_df
        if buffer_type=='ns':
            filtered_df = df[df['state'].apply(lambda x: x[3] > -0.1 or x[0] >-1)]
            return filtered_df
    # Data removal for minigrid environment
    if 'Lava' in env_name:
        filtered_df = df[~df['state'].apply(lambda x:  x[68] == np.float32(0.2))]
        print(f'---------len----{len(df)}--------{len(filtered_df)}')
        return filtered_df
    if 'Dynamic' in env_name:
        filtered_df = df[~df['state'].apply(lambda x: x[68] == np.float32(0.2))]
        # print(f'---------len----{len(df)}--------{len(filtered_df)}')
        return filtered_df
    return df