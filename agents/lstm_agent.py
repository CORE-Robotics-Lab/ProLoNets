import torch
import torch.nn as nn
from torch.distributions import Categorical
import sys
sys.path.insert(0, '../')
from opt_helpers import replay_buffer, ppo_update
import copy

use_gpu = False


class BaselineLSTMNet(nn.Module):
    def __init__(self, input_dim, output_dim, is_value=False):
        super(BaselineLSTMNet, self).__init__()
        self.num_layers = 1
        self.input_dim = input_dim
        self.batch_size = 1
        self.hidden_dim = input_dim
        self.lin1 = nn.Linear(input_dim, input_dim)
        self.rnn = nn.LSTM(input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.lin2 = nn.Linear(self.hidden_dim, input_dim)
        self.lin3 = nn.Linear(input_dim, output_dim)
        self.sig = nn.ReLU()
        if input_dim > 10:
            self.lin1 = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
            )
        self.softmax = nn.Softmax(dim=2)
        self.is_value = is_value

    def init_hidden(self, batch_size=1):
        """
        Initialize hidden state so that it can be clean for each new series of inputs
        :return: Variable of zeros of shape (num_layers, minibatch_size, hidden_dim)
        """
        first_dim = self.num_layers
        second_dim = batch_size
        self.batch_size = batch_size
        third_dim = self.hidden_dim
        if use_gpu:
            return (torch.zeros(first_dim, second_dim, third_dim).cuda(),
                    torch.zeros(first_dim, second_dim, third_dim).cuda())
        else:
            return (torch.zeros(first_dim, second_dim, third_dim),
                    torch.zeros(first_dim, second_dim, third_dim))

    def forward(self, input_data, hidden_state):
        act_out = self.sig(self.lin1(input_data))
        act_out, hidden_out = self.rnn(act_out, hidden_state)
        act_out = self.lin3(self.sig(self.lin2(self.sig(act_out))))
        if self.is_value:
            return act_out, hidden_out
        else:
            return self.softmax(act_out), hidden_out


class LSTMNet:
    def __init__(self,
                 bot_name='LSTMNet',
                 input_dim=4,
                 output_dim=2,
                 epsilon=0.9,
                 epsilon_decay=0.95,
                 epsilon_min=0.05):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.bot_name = bot_name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.replay_buffer = replay_buffer.ReplayBufferSingleAgent()
        self.action_network = BaselineLSTMNet(input_dim=input_dim,
                                              output_dim=output_dim,
                                              is_value=False)
        self.value_network = BaselineLSTMNet(input_dim=input_dim,
                                             output_dim=output_dim,
                                             is_value=True)
        self.ppo = ppo_update.PPO([self.action_network, self.value_network], two_nets=True)
        self.actor_opt = torch.optim.RMSprop(self.action_network.parameters(), lr=1e-5)
        self.value_opt = torch.optim.RMSprop(self.value_network.parameters(), lr=1e-5)

        self.last_state = [0, 0, 0, 0]
        self.last_action = 0
        self.last_action_probs = torch.Tensor([0])
        self.last_value_pred = torch.Tensor([[0, 0]])
        self.action_hidden_state = None
        self.full_probs = None
        self.value_hidden_state = None
        self.new_action_hidden = None
        self.new_value_hidden = None
        self.reward_history = []
        self.num_steps = 0

    def get_action(self, observation):
        if self.action_hidden_state is None:
            self.action_hidden_state = self.action_network.init_hidden()
            self.value_hidden_state = self.value_network.init_hidden()
            self.new_action_hidden = self.action_network.init_hidden()
            self.new_value_hidden = self.value_network.init_hidden()
        with torch.no_grad():
            obs = torch.Tensor(observation)
            obs = obs.view(1, 1, -1)
            self.last_state = obs
            probs, action_hidden = self.action_network(obs, self.action_hidden_state)
            value_pred, value_hidden = self.value_network(obs, self.value_hidden_state)
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
            self.new_action_hidden = action_hidden
            self.new_value_hidden = value_hidden
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
                                  recurrent_hidden_states=(self.action_hidden_state, self.value_hidden_state),
                                  action_log_probs=self.last_action_probs,
                                  value_preds=self.last_value_pred[self.last_action.item()],
                                  last_action=self.last_action,
                                  full_probs_vector=self.full_probs,
                                  rewards=reward)
        self.action_hidden_state = self.new_action_hidden
        self.value_hidden_state = self.new_value_hidden
        return True

    def end_episode(self, timesteps, num_processes=1):
        self.reward_history.append(timesteps)
        value_loss, action_loss = self.ppo.batch_updates(self.replay_buffer, self)
        bot_name = '../txts/' + self.bot_name + str(num_processes) + '_processes'
        # with open(bot_name + "_losses.txt", "a") as myfile:
        #     myfile.write(str(value_loss + action_loss) + '\n')
        with open(bot_name + '_rewards.txt', 'a') as myfile:
            myfile.write(str(timesteps) + '\n')
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)

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
        fn = fn+self.bot_name+'.pth.tar'
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
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
        }

    def __setstate__(self, state):
        self.action_network = copy.deepcopy(state['action_network'])
        self.value_network = copy.deepcopy(state['value_network'])
        self.ppo = copy.deepcopy(state['ppo'])
        self.actor_opt = copy.deepcopy(state['actor_opt'])
        self.value_opt = copy.deepcopy(state['value_opt'])
        self.epsilon_min = copy.deepcopy(state['epsilon_min'])
        self.epsilon_decay = copy.deepcopy(state['epsilon_decay'])
        self.epsilon = copy.deepcopy(state['epsilon'])

    def duplicate(self):
        new_agent = LSTMNet(bot_name=self.bot_name,
                            input_dim=self.input_dim,
                            output_dim=self.output_dim,
                            epsilon=self.epsilon,
                            epsilon_decay=self.epsilon_decay,
                            epsilon_min=self.epsilon_min)
        new_agent.__setstate__(self.__getstate__())
        return new_agent
