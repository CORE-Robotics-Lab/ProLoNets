import random
import sys
sys.path.insert(0, '../')

class CartPoleHeuristic:
    def __init__(self,
                 bot_name='CartPoleHeuristic',
                 params=None):
        self.bot_name = '../txts/'+bot_name
        from opt_helpers import replay_buffer
        self.replay_buffer = replay_buffer.ReplayBufferSingleAgent()

    def get_action(self, observation):
        ## THIS IS THE REAL PROLO INIT
        action = random.choice([0, 1])
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
        return action

    def end_episode(self, timesteps=0, num_procs=None):
        with open(self.bot_name + '_rewards.txt', 'a') as myfile:
            myfile.write(str(timesteps) + '\n')
        self.reset()

    def save(self, fn=None):
        pass

    def save_reward(self, reward):
        self.replay_buffer.insert(value_preds=10,
                                  rewards=reward)
        return True

    def reset(self):
        self.replay_buffer.clear()

    def duplicate(self):
        return self


class LunarHeuristic:
    def __init__(self,
                 bot_name='Lunar_Heuristic'):
        self.bot_name = '../txts/'+bot_name
        from opt_helpers import replay_buffer
        self.replay_buffer = replay_buffer.ReplayBufferSingleAgent()

    def get_action(self, observation):
        if observation[1] < 1.1:  # 0
            if observation[3] < 0.2:  # 1
                if observation[5] < 0:  # 3
                    if (observation[6] + observation[7]) > 1.2:  # 7
                        action = 0
                    else:
                        if observation[4] > -0.1:  # 11
                            action = 2
                        else:
                            action = 1
                else:
                    action = 3
            else:
                if (observation[6] + observation[7]) > 1.2:  # 4
                    action = 0
                else:
                    if observation[0] > 0.2:  # 8
                        action = 1
                    else:
                        if observation[0] < -0.2:  # 12
                            action = 3
                        else:
                            action = 0
        else:
            if observation[5] > 0.1:  # 2
                if observation[5] < -0.1:  # 5
                    if (observation[6] + observation[7]) > 1.2:  # 9
                        action = 0
                    else:
                        action = 1
                else:
                    if observation[0] > 0.2:  # 10
                        action = 1
                    else:
                        if observation[0] < -0.2:  # 13
                            action = 3
                        else:
                            action = 0
            else:
                if (observation[6] + observation[7]) > 1.2:  # 6
                    action = 0
                else:
                    action = 3
        return action

    def end_episode(self, timesteps=0, num_procs=None):
        with open(self.bot_name + '_rewards.txt', 'a') as myfile:
            myfile.write(str(timesteps) + '\n')
        self.reset()

    def save(self, fn=None):
        pass

    def save_reward(self, reward):
        self.replay_buffer.insert(value_preds=10,
                                  rewards=reward)
        return True

    def reset(self):
        self.replay_buffer.clear()

    def duplicate(self):
        return self


class StarCraftMacroHeuristic:
    def __init__(self,
                 bot_name='SC_Macro_Heuristic'):
        from opt_helpers import replay_buffer
        self.replay_buffer = replay_buffer.ReplayBufferSingleAgent()
        self.bot_name = '../txts/'+bot_name

    def get_action(self, observation):
        if observation[10] + observation[12] > 12:  # Attackers > 12
            if sum(observation[45:65])+sum(observation[82:99])+sum(observation[118:139]) > 4:  # enemy units
                action = 41
            else:
                if sum(observation[65:82])+sum(observation[99:118])+sum(observation[139:157]) > 0:  # enemy buildings
                    action = 39
                else:
                    action = 42
        else:
            if observation[4] > 0.5:  # idle workers
                action = 40
            else:
                if observation[3]-observation[2] < 4:  # low supply
                    action = 1
                else:
                    if observation[9]+observation[157] < 15:  # few probes
                        action = 16
                    else:
                        if observation[30]+observation[178] > 0:  # no assimilators
                            if observation[38]+observation[186] > 1.5:  # stargates 1.5
                                if observation[22]+observation[170] > 7:  # voidrays
                                    action = 39
                                else:
                                    action = 29
                            else:
                                if observation[38] + observation[186] > 0.5:  # stargates 0.5
                                    action = 0
                                else:
                                    action = 10
                        else:
                            if observation[31]+observation[179] > 0.5:  # gateway > 0.5
                                if observation[10]+observation[158] > 6:
                                    if observation[34]+observation[182] > 0.5:
                                        action = 2
                                    else:
                                        action = 6
                                else:
                                    action = 17
                            else:
                                action = 3
        return action

    def end_episode(self, timesteps=0, num_procs=None):
        with open(self.bot_name + '_rewards.txt', 'a') as myfile:
            myfile.write(str(timesteps) + '\n')
        self.reset()

    def save(self, fn=None):
        pass

    def save_reward(self, reward):
        self.replay_buffer.insert(value_preds=10,
                                  rewards=reward)
        return True

    def lower_lr(self):
        pass

    def reset(self):
        self.replay_buffer.clear()

    def duplicate(self):
        return self


class StarCraftMicroHeuristic:
    def __init__(self,
                 bot_name='SC_Micro_Heuristic'):
        from opt_helpers import replay_buffer
        self.replay_buffer = replay_buffer.ReplayBufferSingleAgent()
        self.bot_name = '../txts/'+bot_name

    def get_action(self, observation):
        if observation[14] > 0:
            action = 4
        else:
            if observation[1] > 30:
                if -observation[0] > -20:
                    action = 2
                else:
                    if observation[0] > 40:
                        action = 0
                    else:
                        action = 3
            else:
                if observation[1] > 18:
                    action = 2
                else:
                    if -observation[0] > -40:
                        action = 1
                    else:
                        action = 0
        return action

    def end_episode(self, timesteps=0, num_procs=None):
        with open(self.bot_name + '_rewards.txt', 'a') as myfile:
            myfile.write(str(timesteps) + '\n')
        self.reset()

    def save(self, fn=None):
        pass

    def lower_lr(self):
        pass

    def save_reward(self, reward):
        self.replay_buffer.insert(value_preds=10,
                                  rewards=reward)
        return True

    def reset(self):
        self.replay_buffer.clear()

    def duplicate(self):
        return self


class RandomHeuristic:
    def __init__(self,
                 bot_name='RandomBot',
                 action_dim=2):
        self.action_dim = action_dim
        from opt_helpers import replay_buffer
        self.replay_buffer = replay_buffer.ReplayBufferSingleAgent()
        self.bot_name = '../txts/'+bot_name

    def get_action(self, observation):
        return random.randint(0, self.action_dim-1)

    def end_episode(self, timesteps=0, num_procs=None):
        with open(self.bot_name + '_rewards.txt', 'a') as myfile:
            myfile.write(str(timesteps) + '\n')
        self.reset()

    def save(self, fn=None):
        pass

    def save_reward(self, reward):
        self.replay_buffer.insert(value_preds=10,
                                  rewards=reward)
        return True

    def lower_lr(self):
        pass

    def reset(self):
        self.replay_buffer.clear()

    def duplicate(self):
        return self
