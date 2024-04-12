import argparse
import datetime
import os.path
import gym
import gym_rili
import numpy as np
from algos.rili import RILI
from replay_memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter

from algos.sac_agent import SAC
from replay_memory_sac import ReplayMemorySAC

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default="rili-circle-v0")
parser.add_argument('--resume', default="None")
parser.add_argument('--change_partner', type=float, default=0.5) #TODO
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--save_name', default='run_sac_opponent_final_f')
parser.add_argument('--start_eps', type=int, default=300)
parser.add_argument('--num_eps', type=int, default=5000)

parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
# parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
#                     help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size_SAC', type=int, default=50, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                    help='maximum number of steps (default: 10000)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
# parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
#                     help='Steps sampling random actions (default: 10000)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
# parser.add_argument('--cuda', action="store_true",
#                     help='run on CUDA (default: False)')
args = parser.parse_args()


# Environment
env = gym.make(args.env_name)
env.set_params(change_partner=args.change_partner)

# Agent
agent_ego = RILI(env.action_space, env.observation_space.shape[0], env._max_episode_steps)
agent_other = SAC(env.action_space_other,env.observation_space_other.shape[0],args)

# Tensorboard
folder = "runs/" + 'rili-sac' + "/"
writer = SummaryWriter(folder + '{}_{}'.format(args.save_name, datetime.datetime.now().strftime("%m-%d_%H-%M")))

# Memory
memory_agent_ego = ReplayMemory(capacity=args.num_eps, interaction_length=env._max_episode_steps)
memory_agent_other = ReplayMemorySAC(args.replay_size, args.seed)

# Resume Training
if args.resume != "None":
    agent.load_model(args.resume)
    memory.load_buffer(args.resume)
    args.start_eps = 0

z_prev = np.zeros(10)
z = np.zeros(10)
prev_last_state_ego = env.reset()
last_state_ego = env.reset()
state_other = env.reset_opponent()

# Main loop
for i_episode in range(1, args.num_eps+1):

    if len(memory_agent_ego) > 4:
        z = agent_ego.predict_latent(
                        memory_agent_ego.get_steps(memory_agent_ego.position - 4),
                        memory_agent_ego.get_steps(memory_agent_ego.position - 3),
                        memory_agent_ego.get_steps(memory_agent_ego.position - 2),
                        memory_agent_ego.get_steps(memory_agent_ego.position - 1))

    episode_reward_ego = 0
    episode_reward_other = 0
    episode_steps = 0
    done = False
    state_ego = env.reset()
    updates = 0
    # first timestep
    if i_episode < args.start_eps:
        action_agent_other = env.action_space_other.sample()
    else:
        print('select action:',state_other,last_state_ego)
        action_agent_other = agent_other.select_action(state_other,last_state_ego)
    print('Episode ',i_episode)
    while not done:
        if i_episode < args.start_eps:
            action_agent_ego = env.action_space.sample()
        else:
            action_agent_ego = agent_ego.select_action(state_ego, z)

        if len(memory_agent_ego) > args.batch_size:
            critic_1_loss1, critic_2_loss1, policy_loss1, ae_loss1, curr_loss1, next_loss1, kl_loss1 = agent_ego.update_parameters(memory_agent_ego, args.batch_size)
            # writer.add_scalar('autoencoder/ae_loss', ae_loss, agent.updates)
            # writer.add_scalar('autoencoder/z_curr_loss', curr_loss, agent.updates)
            # writer.add_scalar('autoencoder/z_next_loss', next_loss, agent.updates)
            # writer.add_scalar('autoencoder/kl_loss', kl_loss, agent.updates)
            # writer.add_scalar('SAC/critic_1', critic_1_loss, agent.updates)
            # writer.add_scalar('SAC/critic_2', critic_2_loss, agent.updates)
            # writer.add_scalar('SAC/policy', policy_loss, agent.updates)
            
        next_states, rewards, done, _ = env.step([action_agent_ego, action_agent_other])
        reward_ego, reward_other = rewards[0], rewards[1]
        next_state_ego, next_state_other, last_state_ego = next_states[0], next_states[1], next_states[2]
        episode_steps += 1
        episode_reward_ego += reward_ego
        episode_reward_other += reward_other

        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        # print('')
        memory_agent_ego.push_timestep(state_ego, action_agent_ego, reward_ego, next_state_ego, mask)
        state_ego = next_state_ego
        # print('Timestep','state hider',next_state_other,'state ego-seeker',state_ego, 'last_state_ego', last_state_ego)

    # after while loop
    if len(memory_agent_other) > args.batch_size_SAC:
        if np.random.random() > args.change_partner:
            print('Changed partner: update sac')
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss = agent_other.update_parameters(memory_agent_other, args.batch_size_SAC, updates)
                # , ent_loss, alpha 
                updates += 1
    #10th timestep - strictly we took the action in first timestep but we want the final reward from last timestep to be updated in the buffer
    # print('At timestep 10, previous state hider',state_other,'next_state',next_state_other,'reward',reward_other,prev_last_state_ego,last_state_ego)
    memory_agent_other.push(state_other, action_agent_other, reward_other, next_state_other,prev_last_state_ego, last_state_ego, mask)
    state_other = next_state_other
    prev_last_state_ego = last_state_ego

    z_prev = np.copy(z)
    memory_agent_ego.push_interaction()
    #memory_agent_other.push()
    writer.add_scalar('reward/episode_reward', episode_reward_ego, i_episode)
    writer.add_scalar('reward/episode_reward_SAC', episode_reward_other, i_episode)
    print("Episode: {}, partner: {}, reward for Ego/RILI/Seeker: {}".format(i_episode, env.partner, round(episode_reward_ego, 2)))
    print("Hider Agent reward: {}".format( round(episode_reward_other, 2)))

    if i_episode % 1000 == 0:
        agent_ego.save_model(args.save_name + '_' + str(i_episode))
        memory_agent_ego.save_buffer(args.save_name + '_' + str(i_episode))
