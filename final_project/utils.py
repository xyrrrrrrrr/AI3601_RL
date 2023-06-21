'''
This file is used to store our utility functions.
'''
import numpy as np
import argparse
import torch
from copy import deepcopy

def read_data(file_path)-> np.ndarray:
    '''
    This function is used to read the data from the file.
    :param file_path: the path of the file(.npy)
    :return: the data in the file
    '''
    # read the data from the npy file
    return np.load(file_path)

def get_dataset(device)-> list:
    '''
    This function is used to get the dataset from the datadir.
    :param device: the device
    :return: the tensor dataset
    '''
    state = read_data('data/Robust_Hopper-v4_0_10w_state.npy')
    action = read_data('data/Robust_Hopper-v4_0_10w_action.npy')
    reward = read_data('data/Robust_Hopper-v4_0_10w_reward.npy')
    next_state = read_data('data/Robust_Hopper-v4_0_10w_next_state.npy')
    not_done = read_data('data/Robust_Hopper-v4_0_10w_not_done.npy')
    # convert the data to tensor
    state = torch.tensor(state, dtype=torch.float32, device=device)
    action = torch.tensor(action, dtype=torch.float32, device=device)
    reward = torch.tensor(reward, dtype=torch.float32, device=device)
    next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
    not_done = torch.tensor(not_done, dtype=torch.float32, device=device)
    
    return [state, action, reward, next_state, not_done]


def sample(dataset, batch_size=100)-> list:
    '''
    This function is used to sample the dataset.
    :param dataset: the dataset
    :param length: the length of the sample
    :return: the sample
    '''
    # generate the index not repeat
    index = np.random.choice(len(dataset[0]), batch_size, replace=False)
    # sample the dataset
    b_state = dataset[0][index]
    b_action = dataset[1][index]
    b_reward = dataset[2][index]
    b_next_state = dataset[3][index]
    b_not_done = dataset[4][index]

    return [b_state, b_action, b_reward, b_next_state, b_not_done]

def sample_whole_process(dataset, batch_size)->list:
    '''
    Get an whole process from the dataset.
    '''
    # get a random end index
    dataset_here = deepcopy(dataset)
    not_done = dataset_here[4]
    tool = np.where(not_done == 0)[0].tolist()
    tool.append(-1)
    start = 0
    count = 0
    rew = 0
    start_list = []
    end_list = []
    rew_list = []
    for t in tool:
        start = t + 1
        count = start
        rew = dataset_here[2][count]
        while True:
            count += 1  
            if count >= len(not_done) - 1:
                break
            if not_done[count] == 0:
                break
            rew += dataset_here[2][count]
        # print(rew)
        if count - start > batch_size * 2:
            start_list.append(start)
            end_list.append(count)
            rew_list.append(rew)
    b_state = []
    b_action = []
    b_reward = []
    b_next_state = []
    b_not_done = []
    for start,end in zip(start_list, end_list):
        # get the whole process
        b_state.append(dataset_here[0][start:end])
        b_action.append(dataset_here[1][start:end])
        b_reward.append(dataset_here[2][start:end])
        b_next_state.append(dataset_here[3][start:end])
        b_not_done.append(dataset_here[4][start:end])

    return [b_state, b_action, b_reward, b_next_state, b_not_done], rew_list



def make_parser()-> argparse.ArgumentParser:
    '''
    This function is used to make the parser.
    :return: the parser
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Hopper-v2")               # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name", default="Robust")          # Prepends name to filename
    parser.add_argument("--eval_freq", default=200, type=float)     # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment or train for (this defines buffer size)
    parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used before training behavioral
    parser.add_argument("--rand_action_p", default=0.2, type=float) # Probability of selecting random action during batch generation
    parser.add_argument("--gaussian_std", default=0.1, type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
    parser.add_argument("--batch_size", default=200, type=int)      # Mini batch size for networks
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--lmbda", default=0.75)                    # Weighting for clipped double Q-learning in BCQ
    parser.add_argument("--phi", default=0.05)                      # Max perturbation hyper-parameter for BCQ
    args = parser.parse_args()

    return args

def layer_init(layer, std=np.sqrt(2), bias_const=0.0)-> torch.nn.Module:
    '''
    This function is used to initialize the layer.
    '''
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    
    return layer
# smooth the curve
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            # print(previous)
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
if __name__ == '__main__':
    # get dataset
    dataset = get_dataset('cpu')
    # get all the process
    process, rewards = sample_whole_process(dataset, 0)
    for i in range(len(rewards)):
        rewards[i] = rewards[i].item()
    print('mean:', np.mean(rewards), 'std:', np.std(rewards), 'max:', max(rewards), 'min:', min(rewards), 'length',len(rewards))
    import matplotlib.pyplot as plt
    plt.plot(rewards, alpha=0.3, label='original curve')
    plt.plot(smooth_curve(rewards), label='smoothed curve')
    plt.legend()
    plt.show()