import torch
import torch.nn as nn
import sys
sys.path.insert(0, '../')
from torch.distributions import Categorical
from opt_helpers import replay_buffer, ppo_update
import copy
from agents.djinn_helpers import PyDJINN, DJINN_TREE_DATA, tree_to_nn_weights


class DJINNAgent:
    def __init__(self,
                 bot_name='DJINNAgent',
                 input_dim=4,
                 output_dim=2,
                 drop_prob=0.0
                 ):
        self.bot_name = bot_name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.replay_buffer = replay_buffer.ReplayBufferSingleAgent()

        if input_dim == 4:
            self.env_name = 'cart'
        elif input_dim == 6:  # Fire Sim
            self.env_name = 'fire'
        elif input_dim == 8:
            self.env_name = 'lunar'
        elif input_dim == 37:
            self.env_name = 'sc_micro'
        elif input_dim > 100:
            self.env_name = 'sc_macro'
        tree_dict = DJINN_TREE_DATA[self.env_name]
        action_param_dict = tree_to_nn_weights(input_dim, output_dim, tree_dict)
        # value_param_dict = tree_to_nn_weights(input_dim, 1, tree_dict)
        self.action_network = PyDJINN(input_dim,
                                      weights=action_param_dict['weights'],
                                      biases=[], drop_prob=drop_prob,
                                      is_value=False
                                      )
        self.value_network = PyDJINN(input_dim,
                                     weights=action_param_dict['weights'],
                                     biases=[],
                                     drop_prob=drop_prob,
                                     is_value=True
                                     )
        self.ppo = ppo_update.PPO([self.action_network, self.value_network], two_nets=True)
        self.actor_opt = torch.optim.RMSprop(self.action_network.parameters(), lr=5e-3)
        self.value_opt = torch.optim.RMSprop(self.value_network.parameters(), lr=5e-3)

        self.last_state = [0, 0, 0, 0]
        self.last_action = 0
        self.last_action_probs = torch.Tensor([0])
        self.last_value_pred = torch.Tensor([[0, 0]])
        self.last_deep_action_probs = torch.Tensor([0])
        self.last_deep_value_pred = torch.Tensor([[0, 0]])
        self.full_probs = None
        self.reward_history = []
        self.num_steps = 0

    def get_action(self, observation):
        with torch.no_grad():
            obs = torch.Tensor(observation)
            obs = obs.view(1, -1)
            self.last_state = obs
            probs = self.action_network(obs)
            value_pred = self.value_network(obs)
            probs = probs.view(-1)
            self.full_probs = probs
            if self.action_network.input_dim > 30:
                probs, inds = torch.topk(probs, 3)
            m = Categorical(probs)
            action = m.sample()
            # # Epsilon learning
            # action = torch.argmax(probs)
            # if random.random() < self.epsilon:
            #     action = torch.LongTensor([random.choice(np.arange(0, self.output_dim, dtype=np.int))])
            # # End Epsilon learning
            log_probs = m.log_prob(action)
            self.last_action_probs = log_probs
            self.last_value_pred = value_pred.view(-1).cpu()

            if self.action_network.input_dim > 30:
                self.last_action = inds[action]
            else:
                self.last_action = action
        if self.action_network.input_dim > 30:
            action = inds[action].item()
        else:
            action = action.item()
        return action

    def save_reward(self, reward):
        self.replay_buffer.insert(obs=[self.last_state],
                                  action_log_probs=self.last_action_probs,
                                  value_preds=self.last_value_pred[self.last_action.item()],
                                  # value_preds=self.last_value_pred.item(),
                                  last_action=self.last_action,
                                  full_probs_vector=self.full_probs,
                                  rewards=reward)
        return True

    def end_episode(self, timesteps, num_processes=1):
        self.reward_history.append(timesteps)
        value_loss, action_loss = self.ppo.batch_updates(self.replay_buffer, self)
        bot_name = '../txts/' + self.bot_name + str(num_processes) + '_processes'
        self.num_steps += 1
        with open(bot_name + '_rewards.txt', 'a') as myfile:
            myfile.write(str(timesteps) + '\n')

    def lower_lr(self):
        for param_group in self.ppo.actor_opt.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5
        for param_group in self.ppo.critic_opt.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5

    def reset(self):
        self.replay_buffer.clear()

    def deepen_networks(self):
        pass

    def save(self, fn='last'):
        checkpoint = dict()
        checkpoint['actor'] = self.action_network.state_dict()
        checkpoint['value'] = self.value_network.state_dict()
        torch.save(checkpoint, fn+self.bot_name+'.pth.tar')

    def load(self, fn='last'):
        # fn = fn + self.bot_name + '.pth.tar'
        model_checkpoint = torch.load(fn, map_location='cpu')
        actor_data = model_checkpoint['actor']
        value_data = model_checkpoint['value']
        self.action_network.load_state_dict(actor_data)
        self.value_network.load_state_dict(value_data)

    def __getstate__(self):
        return {
            'action_network': self.action_network,
            'value_network': self.value_network,
            'ppo': self.ppo,
            'actor_opt': self.actor_opt,
            'value_opt': self.value_opt,
        }

    def __setstate__(self, state):
        self.action_network = copy.deepcopy(state['action_network'])
        self.value_network = copy.deepcopy(state['value_network'])
        self.ppo = copy.deepcopy(state['ppo'])
        self.actor_opt = copy.deepcopy(state['actor_opt'])
        self.value_opt = copy.deepcopy(state['value_opt'])

    def duplicate(self):
        new_agent = DJINNAgent(
            bot_name=self.bot_name,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
                 )
        new_agent.__setstate__(self.__getstate__())
        return new_agent
