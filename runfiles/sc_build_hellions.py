import math
import sc2
import random
from sc2 import Race, Difficulty
import os
import sys
from sc2.constants import *
from sc2.position import Pointlike, Point2
from sc2.player import Bot, Computer
from sc2.unit import Unit as sc2Unit
sys.path.insert(0, os.path.abspath('../'))
import torch
from agents.prolonet_agent import DeepProLoNet
from agents.py_djinn_agent import DJINNAgent
from agents.lstm_agent import LSTMNet
from agents.baseline_agent import FCNet
from runfiles import sc_build_hellions_helpers
import numpy as np
import time
import torch.multiprocessing as mp
import argparse
from datetime import datetime
import traceback
import random

DEBUG = False
SUPER_DEBUG = False
if SUPER_DEBUG:
    DEBUG = True

FAILED_REWARD = -0.0
STEP_PENALTY = 0.01
SUCCESS_BUILD_REWARD = 0.2
SUCCESS_TRAIN_REWARD = 0.3
SUCCESS_SCOUT_REWARD = 0.
SUCCESS_ATTACK_REWARD = 0.
SUCCESS_MINING_REWARD = 0.01


class StarmniBot(sc2.BotAI):
    def __init__(self, rl_agent):
        super(StarmniBot, self).__init__()
        self.player_unit_tags = []
        self.agent = rl_agent
        self.corners = None
        self.action_buffer = []
        self.prev_state = None
        # self.last_known_enemy_units = []
        self.itercount = 0
        # self.last_scout_iteration = -100
        # self.scouted_bases = []
        self.last_reward = 0
        # self.num_nexi = 1
        self.mining_reward = SUCCESS_MINING_REWARD
        self.last_sc_action = 43  # Nothing
        self.positions_for_depots = []   # replacing with positions for depots
        self.positions_for_buildings = []
        # self.army_below_half = 0
        # self.last_attack_loop = 0
        self.debug_count = 0

    async def on_step(self, iteration):
        if self.units(UnitTypeId.COMMANDCENTER).amount < 1 and self.workers.amount < 1:
            await self._client.leave()
        if iteration == 0:
            # all mineral finding done relative to first command center
            our_center = self.units(UnitTypeId.COMMANDCENTER).first.position
            base_height = self._game_info.terrain_height[self.game_info.map_center.rounded]

            # tilesize given as largest building we would build being a 3x3 and 1 tile wide path for units
            tilesize_x  = 4
            tilesize_y = 3
            #min_x, min_y, max_x, max_y = self.game_info.playable_area
            #hardcoded as burnysc2 clearly has a bug and is returning entirely different values for play area
            min_x, min_y, max_x, max_y = 20, 17, 50, 39
            range_x = max_x - min_x
            range_y = max_y - min_y

            initial_mineral_field_search_radius = 16
            min_pos = self.state.mineral_field.closer_than(initial_mineral_field_search_radius, our_center).center
            vector = min_pos.direction_vector(our_center)
            if vector.x == 0 or vector.y == 0:
                vector = min_pos.direction_vector(our_center)
            #p0 = our_center + (vector * Point2((-3, -3))).rounded  # will have to tune these locations
            # TODO: fix the locations to the true locations. this will require much trial and error
            for i in np.arange(21, 50, 4):
                for j in np.arange(17, 37, 3):
                    pos = Point2((float(i), float(j)))
                    self.positions_for_depots.append(pos)
                    self.positions_for_buildings.append(pos)

            self.positions_for_depots.sort(key=lambda a: our_center.distance_to(a))
            self.positions_for_buildings.sort(key=lambda a: self.start_location.distance_to(a))

            # print(self.positions_for_depots)

            await self.chat_send("ProLo")
            self.corners = [Point2(Pointlike([min_x, max_y])),
                           Point2(Pointlike([min_x, min_y])),
                           Point2(Pointlike([max_x, max_y])),
                           Point2(Pointlike([max_x, min_y]))]

        self.itercount += 1
        # if self.itercount % 10 != 0:
        #     return
        # Get current state (minerals, gas, idles, etc.)
        current_state = sc_build_hellions_helpers.get_player_state(self.state)
        # TODO: Modify the get_player_state to return a state that is more useful to building more marines
        current_state = np.array(current_state)
        my_unit_type_arr = sc_build_hellions_helpers.my_units_to_type_count(self.units)
        # Get pending
        pending = []
        for unit_type in sc_build_hellions_helpers.MY_POSSIBLES:
            if self.already_pending(unit_type):
                pending.append(1)
            else:
                pending.append(0)

        # Reshape all into batch of 1
        my_unit_type_arr = my_unit_type_arr.reshape(-1)  # batch_size, len
        current_state = current_state.reshape(-1)
        #enemy_unit_type_arr = enemy_unit_type_arr.reshape(-1)
        pending = np.array(pending).reshape(-1)
        last_act = np.array([0]).reshape(-1)
        self.prev_state = np.concatenate((current_state,
                                          my_unit_type_arr,
                                          #enemy_unit_type_arr,
                                          pending,
                                          last_act))

        action = self.agent.get_action(self.prev_state)
        # TODO: abstract the act of getting an action into our own agent
        self.last_reward = await self.activate_sub_bot(action)  # delegates the specific action to the baby bots below
        self.last_reward -= STEP_PENALTY
        self.agent.save_reward(self.last_reward)
        self.last_sc_action = action
        try:
            await self.do_actions(self.action_buffer)
        except sc2.protocol.ProtocolError:
            print("Not in game?")
            self.action_buffer = []

            return
        self.action_buffer = []

    async def activate_sub_bot(self, general_bot_out):

        if general_bot_out == 43:
            if SUPER_DEBUG:
                print("Nothing")
            return FAILED_REWARD*0.05

        elif general_bot_out == 11:  # return to mining
            if SUPER_DEBUG:
                print("Mining")
            try:
                # return await self.back_to_mining()
                # first one is idle workers, but what does the second condition mean?
                if self.prev_state[4] > 0:  # what do these states mean?
                    await self.distribute_workers()
                    return SUCCESS_MINING_REWARD
                else:
                    return FAILED_REWARD
                # return self.mining_reward
            except Exception as e:
                print("Mining Exception", e)
                return FAILED_REWARD
            # return await self.distribute_workers()

        else:
            if SUPER_DEBUG:
                print("Building", general_bot_out)
            try:
                return await self.activate_builder(general_bot_out)
                # this is a catch-all that tells it to build a unit of the specified ID.
            except Exception as e:
                print("Builder Exception", e)
                traceback.print_exc()
                return FAILED_REWARD

    async def activate_builder(self, hard_coded_choice):
        """
        call on the builder_bot to choose a unit to create, and select a target location if applicable
        :param hard_coded_choice: output from the RL agent
        :return: reward for action based on legality
        """
        self.debug_count += 1

        unit_choice = hard_coded_choice
        if SUPER_DEBUG:
            print(sc_build_hellions_helpers.my_units_to_str(unit_choice), 'idx:', unit_choice)

        # Need to know unit type to know if placement is valid.
        success = FAILED_REWARD
        index_to_unit = {
            0: UnitTypeId.COMMANDCENTER,
            1: UnitTypeId.SUPPLYDEPOT,
            2: UnitTypeId.REFINERY,
            3: UnitTypeId.BARRACKS,
            4: UnitTypeId.ENGINEERINGBAY,
            5: UnitTypeId.ARMORY,
            6: UnitTypeId.FACTORY,
            7: UnitTypeId.STARPORT
        }  # depending on the application we can narrow this down even further
        if unit_choice < 8:
            # If the builder wants to build a building, we need to place it:
            if unit_choice == 0:
                # Bases will automatically move into expansion slots
                target_pt = None
                # If we just want something to do
            elif unit_choice == 2:
                # Gas collectors will automatically move onto vespene geysers
                target_pt = None
            elif unit_choice == 1:
                positions_for_depots_idx = 0
                target_pt = self.positions_for_depots[positions_for_depots_idx]
                pos_ind = 0
            else:
                positions_for_depots_idx = 0
                target_pt = self.positions_for_depots[positions_for_depots_idx]
                pos_ind = 0
            if target_pt is None:  # the target building is a command center or a refinery
                try:
                    success = await self.build_building(unit_choice, None)
                except Exception as e:
                    if DEBUG:
                        print("exception (build_building None target)", e)
                    success = FAILED_REWARD
            else:  # the target building is not something that automatically snaps to location
                while True:
                    # supply depot
                    if unit_choice == 1 and await self.can_place(index_to_unit[unit_choice], target_pt):
                        break  # TODO: Eventually I'd like to pop(pos_ind) off of the list...
                    elif await self.can_place(index_to_unit[unit_choice], target_pt):
                        break  # TODO: Eventually I'd like to pop(pos_ind) off of the list...
                    else: # failed to place building?

                        positions_for_depots_idx += 1
                        if positions_for_depots_idx >= len(self.positions_for_depots):
                            print("I can't build that!")
                            break
                        target_pt = self.positions_for_depots[positions_for_depots_idx]
                        # self.positions_for_depots.pop(positions_for_depots_idx)
                try:
                    # If can place and psionic covers (or is pylon)
                    success = await self.build_building(unit_choice, target_pt)
                except Exception as e:
                    if DEBUG:
                        print("Exception (build_building)", e)
                    success = FAILED_REWARD
        elif unit_choice < 34:  # not a building but a unit
            try:
                success = await self.train_unit(unit_choice)
            except Exception as e:
                if DEBUG:
                    print("Exception (training)", e)
                success = FAILED_REWARD
        elif unit_choice < 39:  # not a building or a unit but an upgrade
            try:
                success = await self.research_upgrade(unit_choice)
            except Exception as e:
                if DEBUG:
                    print("Exception (research upgrade)", e)
                success = FAILED_REWARD
        return success

    async def build_building(self, action_building_index, placement_location=None):
        """
            This function takes in an action index from the RL agent and maps it to a building to build. For now, these are
            totally hand engineered. Hash. Tag. Deal.
            :param action_building_index: index of action from the RL agent
            :param placement_location: where to drop the new building
            :return: reward for action based on legality
            """
        action_indices_to_buildings = {
            0: UnitTypeId.COMMANDCENTER,
            1: UnitTypeId.SUPPLYDEPOT,
            2: UnitTypeId.REFINERY,
            3: UnitTypeId.BARRACKS,
            4: UnitTypeId.ENGINEERINGBAY,
            5: UnitTypeId.ARMORY,
            6: UnitTypeId.FACTORY,
            7: UnitTypeId.STARPORT,
        }
        building_target = action_indices_to_buildings[action_building_index]
        if self.workers.amount < 1:
            return FAILED_REWARD
        if DEBUG:
            if placement_location is not None:
                print("Building a", building_target.name, "at (", placement_location.position.x, ",", placement_location.position.y, ")")
            else:
                print("Building a", building_target.name)
        if self.can_afford(building_target):
            if building_target == UnitTypeId.REFINERY:
                ccs = self.units(UnitTypeId.COMMANDCENTER).ready
                for cc in ccs:
                    geysers_nearby = self.state.vespene_geyser.closer_than(20.0, cc)

                    for geyser in geysers_nearby:
                        worker = self.select_build_worker(geyser.position, force=True)
                        # if there isn't already a refinery in the geyser
                        if not self.units(UnitTypeId.REFINERY).closer_than(1.0, geyser).exists and worker is not None:
                            self.action_buffer.append(worker.build(UnitTypeId.REFINERY, geyser))
                            return SUCCESS_BUILD_REWARD
                return FAILED_REWARD

            else:
                if placement_location is None:
                    pos_dist = random.random()*2 + 3
                    # pos_dist = 0.5
                    pos = placement_location.position.to2.towards(random.choice(self.corners), pos_dist)
                else:
                    pos = placement_location
                worker = self.select_build_worker(pos, force=True)
                if worker is not None:
                    if building_target == UnitTypeId.SUPPLYDEPOT:
                        self.action_buffer.append(worker.build(building_target, pos))
                        return SUCCESS_BUILD_REWARD
                    elif building_target == UnitTypeId.BARRACKS:
                        if self.units(UnitTypeId.SUPPLYDEPOT).amount < 1: # no supply depots, can't make a barracks dummy
                            return FAILED_REWARD
                        else:
                            self.action_buffer.append(worker.build(building_target, pos))
                            return SUCCESS_BUILD_REWARD * 4  # let's encourage higher tech
                    elif building_target == UnitTypeId.FACTORY:
                        if self.units(
                            UnitTypeId.BARRACKS).amount < 1:  # no barracks, can't make a factory, dummy
                            return FAILED_REWARD
                        else:
                            self.action_buffer.append(worker.build(building_target, pos))
                            return SUCCESS_BUILD_REWARD * 16  # to encourage higher tech

                    # print('action buffer len:', len(self.action_buffer))
                    # await self.build(building_target, near=pos)
                    # else you picked a building we haven't implemented yet. Sorry, wait for the update?
                    return FAILED_REWARD
                else:
                    return FAILED_REWARD
        return FAILED_REWARD

    async def train_unit(self, action_unit_index):
        action_to_unit = {
            8: UnitTypeId.SCV,
            9: UnitTypeId.MARINE,
            10: UnitTypeId.HELLION
            # 34: UnitTypeId.INTERCEPTOR,  # TRAIN BY DEFAULT, DONT NEED TO TRAIN
            # 35: UnitTypeId.ARCHON,  # CURRENT IMPOSSIBLE WITH THE API
        }
        unit_to_build = action_to_unit[action_unit_index]
        if DEBUG:
            print("Training a", unit_to_build.name)
        barracks_units = [UnitTypeId.MARINE]
        factory_units = [UnitTypeId.HELLION]
        if unit_to_build == UnitTypeId.SCV:
            cc = self.units(UnitTypeId.COMMANDCENTER).ready.random
            if cc.noqueue and self.can_afford(unit_to_build):
                self.action_buffer.append(cc.train(unit_to_build))
                return SUCCESS_TRAIN_REWARD
            else:
                return FAILED_REWARD
        elif unit_to_build in barracks_units:
            # Look for a reactored barracks first
                if self.units(UnitTypeId.BARRACKSREACTOR).ready.exists:
                    # one to one the list of abilities needed to train each unit in the barracks menu
                    abilities_necessary = [AbilityId.BARRACKSTRAIN_MARINE]
                    ability_req = abilities_necessary[barracks_units.index(unit_to_build)]
                    building = self.units(UnitTypeId.BARRACKSREACTOR).ready.random
                    abilities = await self.get_available_abilities(building)
                    # all the units have the same cooldown anyway so let's just look at ZEALOT
                    if ability_req in abilities and self.can_afford(unit_to_build):
                        self.action_buffer.append(building.train(unit_to_build))
                        return SUCCESS_TRAIN_REWARD
                    else:
                        return FAILED_REWARD
                elif self.units(UnitTypeId.BARRACKS).ready.exists:
                    building = self.units(UnitTypeId.BARRACKS).ready.random
                    abilities_necessary = [AbilityId.BARRACKSTRAIN_MARINE]
                    ability_req = abilities_necessary[barracks_units.index(unit_to_build)]
                    abilities = await self.get_available_abilities(building)
                    if ability_req in abilities and self.can_afford(unit_to_build):
                        self.action_buffer.append(building.train(unit_to_build))
                        return SUCCESS_TRAIN_REWARD
                    else:
                        # Can't afford or can't make unit type
                        return FAILED_REWARD
        elif unit_to_build in factory_units:
            # Look for a reactored factory first
                if self.units(UnitTypeId.FACTORYREACTOR).ready.exists:
                    # one to one the list of abilities needed to train each unit in the barracks menu
                    abilities_necessary = [AbilityId.FACTORYTRAIN_HELLION]
                    ability_req = abilities_necessary[factory_units.index(unit_to_build)]
                    building = self.units(UnitTypeId.FACTORYREACTOR).ready.random
                    abilities = await self.get_available_abilities(building)
                    # all the units have the same cooldown anyway so let's just look at ZEALOT
                    if ability_req in abilities and self.can_afford(unit_to_build):
                        self.action_buffer.append(building.train(unit_to_build))
                        if unit_to_build == UnitTypeId.HELLION:
                            return 10
                        return SUCCESS_TRAIN_REWARD
                    else:
                        return FAILED_REWARD
                elif self.units(UnitTypeId.FACTORY).ready.exists:
                    building = self.units(UnitTypeId.FACTORY).ready.random
                    abilities_necessary = [AbilityId.FACTORYTRAIN_HELLION]
                    ability_req = abilities_necessary[factory_units.index(unit_to_build)]
                    abilities = await self.get_available_abilities(building)
                    if ability_req in abilities and self.can_afford(unit_to_build):
                        self.action_buffer.append(building.train(unit_to_build))
                        if unit_to_build == UnitTypeId.HELLION:
                            return 10
                        return SUCCESS_TRAIN_REWARD
                    else:
                        # Can't afford or can't make unit type
                        return FAILED_REWARD

        return FAILED_REWARD
    async def back_to_mining(self):
        reassigned_miners = 0
        for a in self.units(UnitTypeId.REFINERY):
            if a.assigned_harvesters < a.ideal_harvesters:
                w = self.workers.closer_than(20, a)
                if w.exists:
                    self.action_buffer.append(w.random.gather(a))
                    reassigned_miners += 1
        for idle_worker in self.workers.idle:
            mf = self.state.mineral_field.closest_to(idle_worker)
            self.action_buffer.append(idle_worker.gather(mf))
            reassigned_miners += 1
        if reassigned_miners > 0:
            return SUCCESS_MINING_REWARD
        else:
            return FAILED_REWARD

    def finish_episode(self, game_result):
        # TODO: replace this bit with the proper minigame reset protocols to make things FAST
        # The map has a trigger that looks for someone sending the word "reset" exactly in chat
        # await self.chat_send("reset")
        # The reward for a rollout should be proportional to the number of marines present at the end
        print("Game over!")
        rewards_list, advantage_list, deeper_advantage_list = discount_reward(
            self.agent.replay_buffer.rewards_list,
            self.agent.replay_buffer.value_list,
            self.agent.replay_buffer.deeper_value_list)
        self.agent.replay_buffer.rewards_list = rewards_list
        self.agent.replay_buffer.advantage_list = advantage_list
        self.agent.replay_buffer.deeper_advantage_list = deeper_advantage_list
        return self.units(UnitTypeId.HELLION).amount

def discount_reward(reward, value, deeper_value):
    R = 0
    rewards = []
    all_rewards = reward
    reward_sum = sum(all_rewards)
    all_values = value
    deeper_all_values = deeper_value
    # Discount future rewards back to the present using gamma
    advantages = []
    deeper_advantages = []

    for r, v, d_v in zip(all_rewards[::-1], all_values[::-1], deeper_all_values[::-1]):
        R = r + 0.99 * R
        rewards.insert(0, R)
        advantages.insert(0, R - v)
        if d_v is not None:
            deeper_advantages.insert(0, R - d_v)
    advantages = torch.Tensor(advantages)
    rewards = torch.Tensor(rewards)

    if len(deeper_advantages) > 0:
        deeper_advantages = torch.Tensor(deeper_advantages)
        deeper_advantages = (deeper_advantages - deeper_advantages.mean()) / (
                deeper_advantages.std() + torch.Tensor([np.finfo(np.float32).eps]))
        deeper_advantage_list = deeper_advantages.detach().clone().cpu().numpy().tolist()
    else:
        deeper_advantage_list = [None] * len(all_rewards)
    # Scale rewards
    rewards = (rewards - rewards.mean()) / (rewards.std() + torch.Tensor([np.finfo(np.float32).eps]))
    advantages = (advantages - advantages.mean()) / (advantages.std() + torch.Tensor([np.finfo(np.float32).eps]))
    rewards_list = rewards.detach().clone().cpu().numpy().tolist()
    advantage_list = advantages.detach().clone().cpu().numpy().tolist()
    return rewards_list, advantage_list, deeper_advantage_list

def run_episode(q, main_agent, episode):
    result = None
    agent_in = main_agent.duplicate()

    bot = StarmniBot(rl_agent=agent_in)

    try: # TODO: replace this with the correct minigame map and set up the game
        replay_filename = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '_e-' + str(episode) + '_' + str(main_agent.bot_name) + '.SC2REPLAY'
        # Alex: I'm not exactly sure of the providence of this map, but we can put a copy here for now:
        # https://drive.google.com/file/d/1GN8pqpxOFhaS7TujRdnmhnO02POJtAD-/view?usp=sharing
        result = sc2.run_game(sc2.maps.get("BuildBCs"),
                              [Bot(Race.Terran, bot)],
                              realtime=False,
                              save_replay_as=os.path.join('sc_replays', replay_filename))
    except KeyboardInterrupt:
        result = [-1, -1]
    except Exception as e:
        print(str(e))
        print("No worries", e, " carry on please")
    if type(result) == list and len(result) > 1:
        result = result[0]

    reward_sum = bot.finish_episode(result)
    agent_in.replay_buffer.extend(bot.agent.replay_buffer.__getstate__())
    if q is not None:
        try:
            q.put([reward_sum, bot.agent.replay_buffer.__getstate__()])
        except RuntimeError as e:
            print(e)
            return [reward_sum, bot.agent.replay_buffer.__getstate__()]
    return [reward_sum, bot.agent.replay_buffer.__getstate__()]

def main(episodes, agent, num_processes):
    running_reward_array = []
    # lowered = False
    for episode in range(1, episodes+1):
        successful_runs = 0
        master_reward, reward, running_reward = 0, 0, 0
        try:
            returned_object = run_episode(None, main_agent=agent, episode=episode)
            master_reward += returned_object[0]
            running_reward_array.append(returned_object[0])
            agent.replay_buffer.extend(returned_object[1])
            successful_runs += 1
        except MemoryError as e:
            print(e)
            continue
        reward = master_reward / float(successful_runs)
        agent.end_episode(reward, num_processes)
        running_reward = sum(running_reward_array[-100:]) / float(min(100.0, len(running_reward_array)))
        if episode % 10 == 0:
            print(f'Episode {episode}  Last Reward: {reward}  Average Reward: {running_reward}')
            print(f"Running {num_processes} concurrent simulations per episode")
        if episode % 100 == 0:
            agent.save('../models/' + str(episode) + 'th')
            agent.lower_lr()
    return running_reward_array


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent_type", help="architecture of agent to run", type=str, default='prolo')
    parser.add_argument("-e", "--episodes", help="how many episodes", type=int, default=1000)
    parser.add_argument("-p", "--processes", help="how many processes?", type=int, default=1)
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
    USE_GPU = args.gpu  # Applies for 'prolo' only. use gpu? Default false
    VECTORIZED = args.vec  # Applies for 'prolo' vectorized or no? Default false
    RANDOM = args.rand  # Applies for 'prolo' random init or no? Default false
    DEEPEN = args.deep  # Applies for 'prolo' deepen or no? Default false
    torch.set_num_threads(NUM_PROCS)
    dim_in = 32
    dim_out = 12
    bot_name = AGENT_TYPE + '_hellions'
    # mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')
    if AGENT_TYPE == 'prolo':
        policy_agent = DeepProLoNet(distribution='one_hot',
                                    bot_name=bot_name,
                                    input_dim=dim_in,
                                    output_dim=dim_out,
                                    use_gpu=USE_GPU,
                                    vectorized=VECTORIZED,
                                    randomized=RANDOM,
                                    adversarial=ADVERSARIAL,
                                    deepen=DEEPEN)
    elif AGENT_TYPE == 'fc':
        policy_agent = FCNet(input_dim=dim_in,
                             bot_name=bot_name,
                             output_dim=dim_out,
                             sl_init=SL_INIT,
                             num_hidden=1)
    elif AGENT_TYPE == 'lstm':
        policy_agent = LSTMNet(input_dim=dim_in,
                               bot_name=bot_name,
                               output_dim=dim_out)
    elif AGENT_TYPE == 'djinn':
        policy_agent = DJINNAgent(bot_name=bot_name,
                                  input_dim=dim_in,
                                  output_dim=dim_out)
    else:
        raise Exception('No valid network selected')
    main(episodes=NUM_EPS, agent=policy_agent, num_processes=NUM_PROCS)