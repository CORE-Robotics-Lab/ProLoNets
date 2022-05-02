import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class PPO:
    def __init__(self, actor_critic_arr, two_nets=True, use_gpu=False):

        lr = 1e-3
        eps = 1e-5
        self.clip_param = 0.2
        self.ppo_epoch = 32
        self.num_mini_batch = 4
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.use_gpu = use_gpu
        if two_nets:
            self.actor = actor_critic_arr[0]
            self.critic = actor_critic_arr[1]
            if self.actor.input_dim > 100:
                self.actor_opt = optim.RMSprop(self.actor.parameters(), lr=1e-5)
                self.critic_opt = optim.RMSprop(self.critic.parameters(), lr=1e-5)
            elif self.actor.input_dim < 8:
                self.actor_opt = optim.RMSprop(self.actor.parameters(), lr=2e-2)
                self.critic_opt = optim.RMSprop(self.critic.parameters(), lr=2e-2)
            else:
                self.actor_opt = optim.RMSprop(self.actor.parameters(), lr=2e-2)
                self.critic_opt = optim.RMSprop(self.critic.parameters(), lr=2e-2)
        else:
            self.actor = actor_critic_arr
            self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr, eps=eps)
        self.two_nets = two_nets
        self.epoch_counter = 0

    def sl_updates(self, rollouts, agent_in, heuristic_teacher):
        if self.actor.input_dim < 10:
            batch_size = max(rollouts.step // 32, 2)
            num_iters = rollouts.step // batch_size
        else:
            num_iters = 4
            batch_size = 32
        aggregate_actor_loss = 0
        for iteration in range(num_iters):
            total_action_loss = torch.Tensor([0])
            total_value_loss = torch.Tensor([0])
            for b in range(batch_size):
                sample = rollouts.sample()
                if not sample:
                    break
                state = sample['state']
                reward = sample['reward']
                if np.isnan(reward):
                    continue
                new_action_probs = self.actor(*state).view(1, -1)
                new_value = self.critic(*state)
                label = torch.LongTensor([heuristic_teacher.get_action(state[0].detach().clone().data.cpu().numpy()[0])])
                action_loss = torch.nn.functional.cross_entropy(new_action_probs, label)
                new_value = new_value.view(-1, 1)
                new_value = new_value[label]
                reward = torch.Tensor([reward]).view(-1, 1)
                value_loss = F.mse_loss(reward, new_value)

                total_value_loss = total_value_loss + value_loss
                total_action_loss = total_action_loss + action_loss
            if total_value_loss != 0:
                self.critic_opt.zero_grad()
                total_value_loss.backward()
                self.critic_opt.step()
            if total_action_loss != 0:
                self.actor_opt.zero_grad()
                total_action_loss.backward()
                self.actor_opt.step()
            aggregate_actor_loss += total_action_loss.item()
        # aggregate_actor_loss /= float(num_iters*batch_size)
        agent_in.reset()
        return aggregate_actor_loss

    def batch_updates(self, rollouts, agent_in, go_deeper=False):
        if self.actor.input_dim == 8:
            batch_size = max(rollouts.step // 32, 1)
            num_iters = rollouts.step // batch_size
        elif self.actor.input_dim == 4:
            batch_size = max(rollouts.step // 32, 2)
            num_iters = rollouts.step // batch_size
        else:
            num_iters = 4
            batch_size = 32
        total_action_loss = torch.Tensor([0])
        total_value_loss = torch.Tensor([0])
        for iteration in range(num_iters):
            total_action_loss = torch.Tensor([0])
            total_value_loss = torch.Tensor([0])
            if self.use_gpu:
                total_action_loss = total_action_loss.cuda()
                total_value_loss = total_value_loss.cuda()
            if go_deeper:
                deep_total_action_loss = torch.Tensor([0])
                deep_total_value_loss = torch.Tensor([0])
                if self.use_gpu:
                    deep_total_value_loss = deep_total_value_loss.cuda()
                    deep_total_action_loss = deep_total_action_loss.cuda()
            samples = [rollouts.sample() for _ in range(batch_size)]
            samples = [sample for sample in samples if sample != False]
            if len(samples) <= 1:
                continue
            state = torch.cat([sample['state'][0] for sample in samples], dim=0)
            action_probs = torch.Tensor([sample['action_prob'] for sample in samples])
            adv_targ = torch.Tensor([sample['advantage'] for sample in samples])
            reward = torch.Tensor([sample['reward'] for sample in samples])
            old_action_probs = torch.cat([sample['full_prob_vector'].unsqueeze(0) for sample in samples], dim=0)
            if True in np.array(np.isnan(adv_targ).tolist()) or \
                    True in np.array(np.isnan(reward).tolist()) or \
                    True in np.array(np.isnan(old_action_probs).tolist()):
                continue
            action_taken = torch.Tensor([sample['action_taken'] for sample in samples])
            if self.use_gpu:
                action_taken = action_taken.cuda()
                state = state.cuda()
                action_probs = action_probs.cuda()
                old_action_probs = old_action_probs.cuda()
                adv_targ = adv_targ.cuda()
                reward = reward.cuda()
            if samples[0]['hidden_state'] is not None:
                actor_hidden_state_batch0 = torch.cat([sample['hidden_state'][0][0] for sample in samples], dim=1)
                actor_hidden_state_batch1 = torch.cat([sample['hidden_state'][0][1] for sample in samples], dim=1)
                actor_hidden_state = (actor_hidden_state_batch0, actor_hidden_state_batch1)
                critic_hidden_state_batch0 = torch.cat([sample['hidden_state'][1][0] for sample in samples], dim=1)
                critic_hidden_state_batch1 = torch.cat([sample['hidden_state'][1][1] for sample in samples], dim=1)
                critic_hidden_state = (critic_hidden_state_batch0, critic_hidden_state_batch1)

                new_action_probs, _ = self.actor(state, actor_hidden_state)
                new_value, _ = self.critic(state, critic_hidden_state)
                new_value = new_value.squeeze(1)
                new_action_probs = new_action_probs.squeeze(1)
            else:
                new_action_probs = self.actor(state)
                new_value = self.critic(state)

            if go_deeper:
                deep_action_probs = torch.Tensor([sample['deeper_action_prob'] for sample in samples])
                deep_adv = torch.Tensor([sample['deeper_advantage'] for sample in samples])
                deeper_old_probs = torch.cat([sample['deeper_full_prob_vector'].unsqueeze(0) for sample in samples], dim=0)
                if self.use_gpu:
                    deep_action_probs = deep_action_probs.cuda()
                    deeper_old_probs = deeper_old_probs.cuda()
                    deep_adv = deep_adv.cuda()

                new_deep_probs = agent_in.deeper_action_network(state)
                new_deep_vals = agent_in.deeper_value_network(state)
                deep_dist = Categorical(new_deep_probs)
                deeper_probs = deep_dist.log_prob(action_taken)
                deeper_action_indices = [int(action_ind.item()) for action_ind in action_taken]
                deeper_val = new_deep_vals[np.arange(0, len(new_deep_vals)), deeper_action_indices]
                deeper_entropy = deep_dist.entropy().mean() * self.entropy_coef
                # deep_ratio = torch.nn.functional.kl_div(new_deep_probs, old_action_probs, reduction='batchmean').pow(-1)
                # #
                # deep_clipped = torch.clamp(deep_ratio, 1.0 - self.clip_param,
                #                            1.0 + self.clip_param).mul(adv_targ).mul(deeper_probs)
                # #
                # deep_ratio = deep_ratio.mul(adv_targ).mul(deeper_probs)
                # deep_action_loss = -torch.min(deep_ratio, deep_clipped).mean()
                #
                deep_ratio = torch.exp(deeper_probs - deep_action_probs)
                deep_surr1 = deep_ratio * deep_adv
                deep_surr2 = torch.clamp(deep_ratio, 1.0-self.clip_param, 1+self.clip_param) * deep_adv
                deep_action_loss = -torch.min(deep_surr1, deep_surr2).mean()
                deep_total_action_loss = deep_total_action_loss + deep_action_loss - deeper_entropy
                deeper_value_loss = F.mse_loss(reward, deeper_val)

                deep_total_value_loss = deep_total_value_loss + deeper_value_loss
                # Copy over shallow params to deeper network
                for weight_index in range(len(self.actor.layers)):
                    new_act_weight = torch.Tensor(self.actor.layers[weight_index].cpu().data.numpy())
                    new_act_comp = torch.Tensor(self.actor.comparators[weight_index].cpu().data.numpy())

                    if self.use_gpu:
                        new_act_weight = new_act_weight.cuda()
                        new_act_comp = new_act_comp.cuda()

                    agent_in.deeper_action_network.layers[weight_index].data = new_act_weight
                    agent_in.deeper_action_network.comparators[weight_index].data = new_act_comp
                for weight_index in range(len(self.critic.layers)):
                    new_val_weight = torch.Tensor(self.critic.layers[weight_index].cpu().data.numpy())
                    new_val_comp = torch.Tensor(self.critic.comparators[weight_index].cpu().data.numpy())
                    if self.use_gpu:
                        new_val_weight = new_val_weight.cuda()
                        new_val_comp = new_val_comp.cuda()
                    agent_in.deeper_value_network.layers[weight_index].data = new_val_weight
                    agent_in.deeper_value_network.comparators[weight_index].data = new_val_comp

            update_m = Categorical(new_action_probs)
            update_log_probs = update_m.log_prob(action_taken)
            action_indices = [int(action_ind.item()) for action_ind in action_taken]
            new_value = new_value[np.arange(0, len(new_value)), action_indices]
            entropy = update_m.entropy().mean().mul(self.entropy_coef)
            # Fake PPO Updates:
            # ratio = torch.div(update_log_probs, action_probs)
            #
            # ratio = torch.nn.functional.kl_div(new_action_probs, old_action_probs, reduction='batchmean').pow(-1)
            # # #
            # clipped = torch.clamp(ratio, 1.0 - self.clip_param,
            #                       1.0 + self.clip_param).mul(adv_targ).mul(update_log_probs)
            # # #
            # ratio = ratio.mul(adv_targ).mul(update_log_probs)
            # action_loss = -torch.min(ratio, clipped).mean()
            #
            # Real PPO Updates
            ratio = torch.exp(update_log_probs - action_probs)
            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
            action_loss = -torch.min(surr1, surr2).mean()
            # Policy Gradient:
            # action_loss = (torch.sum(torch.mul(update_log_probs, adv_targ).mul(-1), -1))
            value_loss = F.mse_loss(reward, new_value)

            total_value_loss = total_value_loss.add(value_loss)
            total_action_loss = total_action_loss.add(action_loss).sub(entropy)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_opt.zero_grad()
            total_value_loss.backward()
            self.critic_opt.step()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_opt.zero_grad()
            total_action_loss.backward()
            self.actor_opt.step()
            if go_deeper:
                nn.utils.clip_grad_norm_(agent_in.deeper_value_network.parameters(), self.max_grad_norm)
                agent_in.deeper_value_opt.zero_grad()
                deep_total_value_loss.backward()
                agent_in.deeper_value_opt.step()
                nn.utils.clip_grad_norm_(agent_in.deeper_action_network.parameters(), self.max_grad_norm)
                agent_in.deeper_actor_opt.zero_grad()
                deep_total_action_loss.backward()
                agent_in.deeper_actor_opt.step()
        agent_in.deepen_networks()
        agent_in.reset()
        self.epoch_counter += 1
        return total_action_loss.item(), total_value_loss.item()
