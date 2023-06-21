'''
This file is used to train the agent.
'''
import gym
import numpy as np
import os
os.add_dll_directory(os.environ["USERPROFILE"] + "/.mujoco/mjpro150/bin")
import torch
import sys
import tqdm
from model import BCQ, AWR
from utils import get_dataset, make_parser

# set C:\Users\28419\.mujoco\mujoco210\bin to PATH

sys.path.append('C:\\Users\\28419\\.mujoco\\mujoco210\\bin')

def train(state_dim, action_dim, max_action, device, args, agent)->None:
    '''
    This function is used to train the agent offline.
    '''
    setting = f"{args.env}_{args.seed}_{args.discount}_{args.tau}_{args.lmbda}_{args.phi}"
    # Initialize policy
    if agent == 'BCQ':
        policy = BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)
        policy.load_pth()
    elif agent == 'AWR':
        policy = AWR(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)
        # policy.load_pth()
        torch.manual_seed(24)
        # np.random.seed(366)
    # load dataset
    dataset = get_dataset(device)

    evaluations = []
    episode_num = 0
    done = True 
    best_reward = 3300
    training_iters = 0
    print("Training...")
    while training_iters < args.max_timesteps: 
        pol_vals = policy.train(dataset=dataset, iterations=int(args.eval_freq), batch_size=args.batch_size)
        print('hihere')
        seed = args.seed + 24
        # seed = args.seed
        avg_reward = eval_policy(policy, args.env, seed)
        # policy.set_current_reward(avg_reward)
        if avg_reward > best_reward:
            break
        evaluations.append(avg_reward)
        # np.save(f"./results/BCQ_{setting}", evaluations)
        training_iters += args.eval_freq
        print(f"Training iterations: {training_iters}")
    print("Training finished.")
    # save the model
    if agent == 'BCQ':
        torch.save(policy.actor.state_dict(), f"./models/BCQ_actor_{setting}_3300.pth")
        torch.save(policy.critic.state_dict(), f"./models/BCQ_critic_{setting}_3300.pth")
        torch.save(policy.vae.state_dict(), f"./models/BCQ_vae_{setting}_3300.pth")
    elif agent == 'AWR':
        torch.save(policy.actor.state_dict(), f"./models/AWR_actor_{setting}_2800.pth")
        torch.save(policy.critic.state_dict(), f"./models/AWR_critic_{setting}_2800.pth")
        # torch.save(policy.vae.state_dict(), f"./models/AWR_vae_{setting}.pth")

def eval_policy(policy, env_name, seed, eval_episodes=32):
	'''
	This function is used to evaluate the policy.
    '''
	eval_env = gym.make(env_name)
	eval_env.seed(seed)

	rewards = []
	for _ in tqdm.tqdm(range(eval_episodes)):
		state, done = eval_env.reset(), False
		cur_reward = 0.       
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			cur_reward += reward
		rewards.append(cur_reward)

	avg_reward = np.mean(rewards)

	print("="*40)
	print(f'rewards: {rewards}')    
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print(f"Max reward: {np.max(rewards):.3f}")
	print(f"Min reward: {np.min(rewards):.3f}")
	print("="*40)
	return avg_reward


if __name__ == '__main__':
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./models"):
        os.makedirs("./models")
    args = make_parser()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed + 1)
    np.random.seed(args.seed + 1)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print(f"state_dim: {state_dim}, action_dim: {action_dim}, max_action: {max_action}")
    train(state_dim, action_dim, max_action, device, args, agent = 'BCQ')