import gym
import numpy as np
import sys
import torch
sys.path.insert(0, '../')
from agents.prolonet_agent import DeepProLoNet
# from agents.non_deep_prolonet_agent import ProLoLoki
# from agents.random_prolonet_agent import RandomProLoNet
from agents.heuristic_agent import LunarHeuristic, CartPoleHeuristic, RandomHeuristic
from agents.lstm_agent import LSTMNet
from agents.baseline_agent import FCNet
from agents.py_djinn_agent import DJINNAgent
import random
from opt_helpers.replay_buffer import discount_reward
import torch.multiprocessing as mp
import argparse
import copy


def run_episode(q, agent_in, ENV_NAME):
    agent = agent_in.duplicate()
    if ENV_NAME == 'lunar':
        env = gym.make('LunarLander-v2')
    elif ENV_NAME == 'cart':
        env = gym.make('CartPole-v1')
    else:
        raise Exception('No valid environment selected')

    state = env.reset()  # Reset environment and record the starting state
    done = False

    while not done:
        action = agent.get_action(state)
        # Step through environment using chosen action
        state, reward, done, _ = env.step(action)
        # Save reward
        agent.save_reward(reward)
        if done:
            break
    reward_sum = np.sum(agent.replay_buffer.rewards_list)
    rewards_list, advantage_list, deeper_advantage_list = discount_reward(agent.replay_buffer.rewards_list,
                                                                          agent.replay_buffer.value_list,
                                                                          agent.replay_buffer.deeper_value_list)
    agent.replay_buffer.rewards_list = rewards_list
    agent.replay_buffer.advantage_list = advantage_list
    agent.replay_buffer.deeper_advantage_list = deeper_advantage_list

    to_return = [reward_sum, copy.deepcopy(agent.replay_buffer.__getstate__())]
    if q is not None:
        try:
            q.put(to_return)
        except RuntimeError as e:
            print(e)
            return to_return
    return to_return


def main(episodes, agent, num_processes, ENV_NAME):
    running_reward_array = []
    for episode in range(episodes):
        master_reward = 0
        reward, running_reward = 0, 0
        processes = []
        # q = mp.Manager().Queue()
        # for proc in range(num_processes):
        #     p = mp.Process(target=run_episode, args=(q, agent, ENV_NAME))
        #     p.start()
        #     processes.append(p)
        # for p in processes:
        #     p.join()
        # while not q.empty():
        #     fake_out = q.get()
        #     master_reward += fake_out[0]
        #     running_reward_array.append(fake_out[0])
        #     agent.replay_buffer.extend(fake_out[1])
        returned_object = run_episode(None, agent_in=agent, ENV_NAME=ENV_NAME)
        master_reward += returned_object[0]
        running_reward_array.append(returned_object[0])
        agent.replay_buffer.extend(returned_object[1])

        reward = master_reward / float(num_processes)
        if reward >= 499:
            agent.save('../models/'+str(episode)+'th')
        agent.end_episode(reward, num_processes)

        running_reward = sum(running_reward_array[-100:]) / float(min(100.0, len(running_reward_array)))
        if episode % 50 == 0:
            print(f'Episode {episode}  Last Reward: {reward}  Average Reward: {running_reward}')
        if episode % 500 == 0:
            agent.save('../models/'+str(episode)+'th')

    return running_reward_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent_type", help="architecture of agent to run", type=str, default='djinn')
    parser.add_argument("-e", "--episodes", help="how many episodes", type=int, default=1000)
    parser.add_argument("-p", "--processes", help="how many processes?", type=int, default=1)
    parser.add_argument("-env", "--env_type", help="environment to run on", type=str, default='cart')
    parser.add_argument("-gpu", help="run on GPU?", action='store_true')
    parser.add_argument("-vec", help="Vectorized ProLoNet?", action='store_true')
    parser.add_argument("-adv", help="Adversarial ProLoNet?", action='store_true')
    parser.add_argument("-rand", help="Random ProLoNet?", action='store_true')
    parser.add_argument("-deep", help="Deepening?", action='store_true')
    parser.add_argument("-s", "--sl_init", help="sl to rl for fc net?", action='store_true')

    args = parser.parse_args()
    AGENT_TYPE = args.agent_type  # 'shallow_prolo', 'prolo', 'random', 'fc', 'lstm'
    ADVERSARIAL = args.adv  # Adversarial prolo, applies for AGENT_TYPE=='shallow_prolo'
    SL_INIT = args.sl_init  # SL->RL fc, applies only for AGENT_TYPE=='fc'
    NUM_EPS = args.episodes  # num episodes Default 1000
    NUM_PROCS = args.processes  # num concurrent processes Default 1
    ENV_TYPE = args.env_type  # 'cart' or 'lunar' Default 'cart'
    USE_GPU = args.gpu  # Applies for 'prolo' only. use gpu? Default false
    VECTORIZED = args.vec  # Applies for 'prolo' vectorized or no? Default false
    RANDOM = args.rand  # Applies for 'prolo' random init or no? Default false
    DEEPEN = args.deep  # Applies for 'prolo' deepen or no? Default false
    EPSILON = 1.0  # Chance to take random action
    EPSILON_DECAY = 0.99  # Multiplicative decay on epsilon
    EPSILON_MIN = 0.1  # Floor for epsilon
    torch.set_num_threads(NUM_PROCS)
    for NUM_PROCS in [NUM_PROCS]:
        if ENV_TYPE == 'lunar':
            init_env = gym.make('LunarLander-v2')
            dim_in = init_env.observation_space.shape[0]
            dim_out = init_env.action_space.n
        elif ENV_TYPE == 'cart':
            init_env = gym.make('CartPole-v1')
            dim_in = init_env.observation_space.shape[0]
            dim_out = init_env.action_space.n
        else:
            raise Exception('No valid environment selected')

        print(f"Agent {AGENT_TYPE} on {ENV_TYPE} with {NUM_PROCS} runners")
        # mp.set_start_method('spawn')
        mp.set_sharing_strategy('file_system')
        for i in range(5):
            bot_name = AGENT_TYPE + ENV_TYPE
            if USE_GPU:
                bot_name += 'GPU'

            if AGENT_TYPE == 'prolo':
                policy_agent = DeepProLoNet(distribution='one_hot',
                                            bot_name=bot_name,
                                            input_dim=dim_in,
                                            output_dim=dim_out,
                                            use_gpu=USE_GPU,
                                            vectorized=VECTORIZED,
                                            randomized=RANDOM,
                                            adversarial=ADVERSARIAL,
                                            deepen=DEEPEN,
                                            epsilon=EPSILON,
                                            epsilon_decay=EPSILON_DECAY,
                                            epsilon_min=EPSILON_MIN
                                            )
            elif AGENT_TYPE == 'fc':
                policy_agent = FCNet(input_dim=dim_in,
                                     bot_name=bot_name,
                                     output_dim=dim_out,
                                     sl_init=SL_INIT)

            elif AGENT_TYPE == 'lstm':
                policy_agent = LSTMNet(input_dim=dim_in,
                                       bot_name=bot_name,
                                       output_dim=dim_out)
            elif AGENT_TYPE == 'random':
                policy_agent = RandomHeuristic(bot_name=bot_name,
                                               action_dim=dim_out)
            elif AGENT_TYPE == 'heuristic':
                if ENV_TYPE == 'lunar':
                    policy_agent = LunarHeuristic(bot_name=bot_name)
                elif ENV_TYPE == 'cart':
                    policy_agent = CartPoleHeuristic(bot_name=bot_name)
            elif AGENT_TYPE == 'djinn':
                policy_agent = DJINNAgent(bot_name=bot_name,
                                          input_dim=dim_in,
                                          output_dim=dim_out)
            else:
                raise Exception('No valid network selected')

            if SL_INIT and i == 0:
                NUM_EPS += policy_agent.action_loss_threshold
            num_procs = NUM_PROCS
            reward_array = main(NUM_EPS, policy_agent, num_procs, ENV_TYPE)
