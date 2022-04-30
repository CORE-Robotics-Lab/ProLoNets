import torch
from torch.distributions import Categorical
import sys
sys.path.insert(0, '../')
from opt_helpers import replay_buffer, ppo_update
from agents.vectorized_prolonet_helpers import init_cart_nets, swap_in_node, add_level, \
    init_lander_nets, init_micro_net, init_adversarial_net, init_sc_nets, init_sc_build_marines_net, \
    init_sc_build_hellions_net, save_prolonet, load_prolonet, init_fire_nets
import copy
import os
from runfiles.build_marines_helpers import TYPES

from runfiles.build_marines_helpers import TYPES

class DeepProLoNet:
    def __init__(self,
                 distribution='one_hot',
                 bot_name='ProLoNet',
                 input_dim=4,
                 output_dim=2,
                 use_gpu=False,
                 vectorized=False,
                 randomized=False,
                 adversarial=False,
                 deepen=True,
                 epsilon=0.9,
                 epsilon_decay=0.95,
                 epsilon_min=0.05,
                 deterministic=False):
        self.replay_buffer = replay_buffer.ReplayBufferSingleAgent()
        self.bot_name = bot_name
        self.use_gpu = use_gpu
        self.vectorized = vectorized
        self.randomized = randomized
        self.adversarial = adversarial
        self.deepen = deepen
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.adv_prob = .05
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.deterministic = deterministic
        self.lr = 2e-2
        if vectorized:
            self.bot_name += '_vect'
        if randomized:
            self.bot_name += '_rand'
        if use_gpu:
            self.bot_name += '_gpu'
        if deepen:
            self.bot_name += '_deepening'
        if input_dim == 4 and output_dim == 2:  # CartPole
            self.action_network, self.value_network = init_cart_nets(distribution, use_gpu, vectorized, randomized)
            if adversarial:
                self.action_network, self.value_network = init_adversarial_net(adv_type='cart',
                                                                               distribution_in=distribution,
                                                                               adv_prob=self.adv_prob)
                self.bot_name += '_adversarial' + str(self.adv_prob)
        elif input_dim == 8 and output_dim == 4:  # Lunar Lander
            self.lr = 2e-2
            self.action_network, self.value_network = init_lander_nets(distribution, use_gpu, vectorized, randomized)
            if adversarial:
                self.action_network, self.value_network = init_adversarial_net(adv_type='lunar',
                                                                               distribution_in=distribution,
                                                                               adv_prob=self.adv_prob)
                self.bot_name += '_adversarial' + str(self.adv_prob)
        elif input_dim == 194 and output_dim == 44:  # SC Macro
            self.action_network, self.value_network = init_sc_nets(distribution, use_gpu, vectorized, randomized)
        elif input_dim == 37 and output_dim == 10:  # SC Micro
            self.action_network, self.value_network = init_micro_net(distribution, use_gpu, vectorized, randomized)
            if adversarial:
                self.action_network, self.value_network = init_adversarial_net(adv_type='micro',
                                                                               distribution_in=distribution,
                                                                               adv_prob=self.adv_prob)
                self.bot_name += '_adversarial' + str(self.adv_prob)
        elif input_dim == 30 and output_dim == 10:   # SC BuildMarines
            self.action_network, self.value_network = init_sc_build_marines_net(distribution, use_gpu, vectorized, randomized)
        elif input_dim == 32 and output_dim == 12:   # SC Build Hellions
            self.action_network, self.value_network = init_sc_build_hellions_net(distribution, use_gpu, vectorized, randomized)
        elif input_dim == 6 and output_dim == 5:  # Fire Sim
            self.action_network, self.value_network = init_fire_nets(distribution,
                                                                     use_gpu,
                                                                     vectorized,
                                                                     randomized,
                                                                     bot_name.split('_')[0])

        self.ppo = ppo_update.PPO([self.action_network, self.value_network], two_nets=True, use_gpu=use_gpu)
        self.actor_opt = torch.optim.RMSprop(self.action_network.parameters(), lr=1e-5)
        self.value_opt = torch.optim.RMSprop(self.value_network.parameters(), lr=1e-5)

        if self.deepen:
            self.deeper_action_network = add_level(self.action_network, use_gpu=use_gpu)
            self.deeper_value_network = add_level(self.value_network, use_gpu=use_gpu)

            self.deeper_actor_opt = torch.optim.RMSprop(self.deeper_action_network.parameters(), lr=self.lr)
            self.deeper_value_opt = torch.optim.RMSprop(self.deeper_value_network.parameters(), lr=self.lr)
        else:
            self.deeper_value_network = None
            self.deeper_action_network = None
            self.deeper_actor_opt = None
            self.deeper_value_opt = None
        self.num_times_deepened = 0
        self.last_state = [0, 0, 0, 0]
        self.last_action = 0
        self.last_action_probs = torch.Tensor([0])
        self.last_value_pred = torch.Tensor([[0, 0]])
        self.last_deep_action_probs = None
        self.last_deep_value_pred = [None]*output_dim
        self.full_probs = None
        self.deeper_full_probs = None
        self.reward_history = []
        self.num_steps = 0



    def get_action(self, observation):
        with torch.no_grad():
            obs = torch.Tensor(observation)
            obs = obs.view(1, -1)
            self.last_state = obs
            if self.use_gpu:
                obs = obs.cuda()
            probs = self.action_network(obs)
            value_pred = self.value_network(obs)
            probs = probs.view(-1).cpu()
            self.full_probs = probs
            # print('action probs:', probs)
            if self.action_network.input_dim >= 30:
                probs, inds = torch.topk(probs, 5)
            m = Categorical(probs)
            action = m.sample()
            if self.deterministic:
                action = torch.argmax(probs)

            log_probs = m.log_prob(action)
            self.last_action_probs = log_probs.cpu()
            self.last_value_pred = value_pred.view(-1).cpu()

            if self.deepen:
                deeper_probs = self.deeper_action_network(obs)
                deeper_value_pred = self.deeper_value_network(obs)
                deeper_probs = deeper_probs.view(-1).cpu()
                self.deeper_full_probs = deeper_probs
                if self.action_network.input_dim >= 30:
                    deeper_probs, _ = torch.topk(probs,5)
                deep_m = Categorical(deeper_probs)
                deep_log_probs = deep_m.log_prob(action)
                self.last_deep_action_probs = deep_log_probs.cpu()
                self.last_deep_value_pred = deeper_value_pred.view(-1).cpu()
            if self.action_network.input_dim >= 30:
                self.last_action = inds[action].cpu()
            else:
                self.last_action = action.cpu()
        if self.action_network.input_dim >= 30:
            action = inds[action].item()
        else:
            action = action.item()
        return action

    def save_reward(self, reward):
        self.replay_buffer.insert(obs=[self.last_state],
                                  action_log_probs=self.last_action_probs,
                                  value_preds=self.last_value_pred[self.last_action.item()],
                                  deeper_action_log_probs=self.last_deep_action_probs,
                                  deeper_value_pred=self.last_deep_value_pred[self.last_action.item()],
                                  last_action=self.last_action,
                                  full_probs_vector=self.full_probs,
                                  deeper_full_probs_vector=self.deeper_full_probs,
                                  rewards=reward)
        return True

    def end_episode(self, timesteps, num_processes):
        value_loss, action_loss = self.ppo.batch_updates(self.replay_buffer, self, go_deeper=self.deepen)
        self.num_steps += 1
        # Copy over new decision node params from shallower network to deeper network
        bot_name = '../txts/' + self.bot_name + str(num_processes) + '_processes'
        with open(bot_name + '_rewards.txt', 'a') as myfile:
            myfile.write(str(timesteps) + '\n')
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)

    def lower_lr(self):
        self.lr = self.lr * 0.5
        for param_group in self.ppo.actor_opt.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5
        for param_group in self.ppo.critic_opt.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5

    def reset(self):
        self.replay_buffer.clear()

    def save(self, fn='last'):
        act_fn = fn + self.bot_name + '_actor_' + '.pth.tar'
        val_fn = fn + self.bot_name + '_critic_' + '.pth.tar'

        deep_act_fn = fn + self.bot_name + '_deep_actor_' + '.pth.tar'
        deep_val_fn = fn + self.bot_name + '_deep_critic_' + '.pth.tar'
        save_prolonet(act_fn, self.action_network)
        save_prolonet(val_fn, self.value_network)
        if self.deepen:
            save_prolonet(deep_act_fn, self.deeper_action_network)
            save_prolonet(deep_val_fn, self.deeper_value_network)

    def load(self, fn='last'):
        act_fn = fn + self.bot_name + '_actor_' + '.pth.tar'
        val_fn = fn + self.bot_name + '_critic_' + '.pth.tar'

        deep_act_fn = fn + self.bot_name + '_deep_actor_' + '.pth.tar'
        deep_val_fn = fn + self.bot_name + '_deep_critic_' + '.pth.tar'
        if os.path.exists(act_fn):
            self.action_network = load_prolonet(act_fn)
            self.value_network = load_prolonet(val_fn)
            if self.deepen:
                self.deeper_action_network = load_prolonet(deep_act_fn)
                self.deeper_value_network = load_prolonet(deep_val_fn)

    def deepen_networks(self):
        if not self.deepen or self.num_times_deepened > 8:
            return
        self.entropy_leaf_checks()
        # Copy over shallow params to deeper network
        for weight_index in range(len(self.action_network.layers)):
            new_act_weight = torch.Tensor(self.action_network.layers[weight_index].cpu().data.numpy())
            new_act_comp = torch.Tensor(self.action_network.comparators[weight_index].cpu().data.numpy())

            if self.use_gpu:
                new_act_weight = new_act_weight.cuda()
                new_act_comp = new_act_comp.cuda()

            self.deeper_action_network.layers[weight_index].data = new_act_weight
            self.deeper_action_network.comparators[weight_index].data = new_act_comp
        for weight_index in range(len(self.value_network.layers)):
            new_val_weight = torch.Tensor(self.value_network.layers[weight_index].cpu().data.numpy())
            new_val_comp = torch.Tensor(self.value_network.comparators[weight_index].cpu().data.numpy())
            if self.use_gpu:
                new_val_weight = new_val_weight.cuda()
                new_val_comp = new_val_comp.cuda()
            self.deeper_value_network.layers[weight_index].data = new_val_weight
            self.deeper_value_network.comparators[weight_index].data = new_val_comp

    def entropy_leaf_checks(self):
        leaf_max = torch.nn.Softmax(dim=0)
        new_action_network = copy.deepcopy(self.action_network)
        changes_made = []
        for leaf_index in range(len(self.action_network.action_probs)):
            existing_leaf = leaf_max(self.action_network.action_probs[leaf_index])
            new_leaf_1 = leaf_max(self.deeper_action_network.action_probs[2*leaf_index+1])
            new_leaf_2 = leaf_max(self.deeper_action_network.action_probs[2*leaf_index])
            existing_entropy = Categorical(existing_leaf).entropy().item()
            new_entropy = Categorical(new_leaf_1).entropy().item() + \
                Categorical(new_leaf_2).entropy().item()

            if new_entropy+0.1 <= existing_entropy:
                with open('../txts/' + self.bot_name + '_entropy_splits.txt', 'a') as myfile:
                    myfile.write('Split at ' + str(self.num_steps) + ' steps' + ': \n')
                    myfile.write('Leaf: ' + str(leaf_index) + '\n')
                    myfile.write('Prior Probs: ' + str(self.action_network.action_probs[leaf_index]) + '\n')
                    myfile.write('New Probs 1: ' + str(self.deeper_action_network.action_probs[leaf_index*2]) + '\n')
                    myfile.write('New Probs 2: ' + str(self.deeper_action_network.action_probs[leaf_index*2+1]) + '\n')

                new_action_network = swap_in_node(new_action_network, self.deeper_action_network, leaf_index, use_gpu=self.use_gpu)
                changes_made.append(leaf_index)
        if len(changes_made) > 0:
            self.action_network = new_action_network

            if self.action_network.input_dim > 100:
                new_actor_opt = torch.optim.RMSprop(self.action_network.parameters(), lr=1e-5)
            elif self.action_network.input_dim >= 8:
                new_actor_opt = torch.optim.RMSprop(self.action_network.parameters(), lr=self.lr)
            else:
                new_actor_opt = torch.optim.RMSprop(self.action_network.parameters(), lr=self.lr)

            self.ppo.actor = self.action_network
            self.ppo.actor_opt = new_actor_opt

            for change in changes_made[::-1]:
                self.num_times_deepened += 1
                self.deeper_action_network = swap_in_node(self.deeper_action_network, None, change*2+1, use_gpu=self.use_gpu)
                self.deeper_action_network = swap_in_node(self.deeper_action_network, None, change*2, use_gpu=self.use_gpu)

            self.deeper_actor_opt = torch.optim.RMSprop(self.deeper_action_network.parameters(), lr=self.lr)

    def __getstate__(self):
        return {
            'action_network': self.action_network,
            'value_network': self.value_network,
            'ppo': self.ppo,
            'deeper_action_network': self.deeper_action_network,
            'deeper_value_network': self.deeper_value_network,
            'actor_opt': self.actor_opt,
            'value_opt': self.value_opt,
            'deeper_actor_opt': self.deeper_actor_opt,
            'deeper_value_opt': self.deeper_value_opt,
            'bot_name': self.bot_name,
            'use_gpu': self.use_gpu,
            'vectorized': self.vectorized,
            'randomized': self.randomized,
            'adversarial':self.adversarial,
            'deepen': self.deepen,
            'output_dim': self.output_dim,
            'input_dim': self.input_dim,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'num_times_deepened': self.num_times_deepened,
            'deterministic': self.deterministic,
        }

    def __setstate__(self, state):
        for key in state:
            setattr(self, key, state[key])

    def duplicate(self):
        new_agent = DeepProLoNet(distribution='one_hot',
                                 bot_name=self.bot_name,
                                 input_dim=self.input_dim,
                                 output_dim=self.output_dim,
                                 use_gpu=self.use_gpu,
                                 vectorized=self.vectorized,
                                 randomized=self.randomized,
                                 adversarial=self.adversarial,
                                 deepen=self.deepen,
                                 epsilon=self.epsilon,
                                 epsilon_decay=self.epsilon_decay,
                                 epsilon_min=self.epsilon_min,
                                 deterministic=self.deterministic
                                 )
        new_agent.__setstate__(self.__getstate__())
        return new_agent
