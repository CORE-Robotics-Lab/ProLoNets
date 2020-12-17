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
from ICML.runfiles import sc_helpers
import numpy as np
import time
import torch.multiprocessing as mp
import argparse

DEBUG = False
SUPER_DEBUG = False
if SUPER_DEBUG:
    DEBUG = True

FAILED_REWARD = -0.0
STEP_PENALTY = 0.01
SUCCESS_BUILD_REWARD = 0.
SUCCESS_TRAIN_REWARD = 0.
SUCCESS_SCOUT_REWARD = 0.
SUCCESS_ATTACK_REWARD = 0.
SUCCESS_MINING_REWARD = 0.


class StarmniBot(sc2.BotAI):
    def __init__(self, rl_agent):
        super(StarmniBot, self).__init__()
        self.player_unit_tags = []
        self.agent = rl_agent
        self.corners = None
        self.action_buffer = []
        self.prev_state = None
        self.last_known_enemy_units = []
        self.itercount = 0
        self.last_scout_iteration = -100
        self.scouted_bases = []
        self.last_reward = 0
        # self.num_nexi = 1
        self.mining_reward = SUCCESS_MINING_REWARD
        self.last_sc_action = 43  # Nothing
        self.positions_for_pylons = []
        self.positions_for_buildings = []
        self.army_below_half = 0
        self.last_attack_loop = 0

    async def on_step(self, iteration):
        if self.units(UnitTypeId.NEXUS).amount < 1 and self.workers.amount < 1:
            await self._client.leave()
        if iteration == 0:
            base_height = self._game_info.terrain_height[self.start_location.rounded]
            max_x = self._game_info.map_size.width
            max_y = self._game_info.map_size.height
            min_pos = self.state.mineral_field.closer_than(8, self.start_location).center
            vector = min_pos.direction_vector(self.start_location)
            if vector.x == 0 or vector.y == 0:
                vector = min_pos.direction_vector(self.game_info.map_center)
            p0 = self.start_location + (vector * Point2((5.5, 5.5))).rounded
            p1 = p0 + vector * Point2((4, 6))
            b0 = p0 + vector * Point2((0.5, 2.5))
            b1 = b0 + vector * Point2((0, 3))
            b2 = p0 + vector * Point2((4.5, 0.5))
            b3 = b2 + vector * Point2((0, 3))
            for y in range(-3, 3):
                for x in range(-3, 3):
                    if x < 0 and y < 0:
                        continue
                    for pos in [p0 + vector * Point2((x * 8, y * 9)), p1 + vector * Point2((x * 8, y * 9))]:
                        if 0 < pos.x < max_x and 0 < pos.y < max_y and \
                                self._game_info.terrain_height[pos.rounded] == base_height:
                            self.positions_for_pylons.append(pos)
                    for pos in [b0 + vector * Point2((x * 8, y * 9)), b1 + vector * Point2((x * 8, y * 9)),
                                b2 + vector * Point2((x * 8, y * 9)), b3 + vector * Point2((x * 8, y * 9))]:
                        if 0 < pos.x < max_x and 0 < pos.y < max_y and \
                                self._game_info.terrain_height[pos.rounded] == base_height:
                            self.positions_for_buildings.append(pos)
            self.positions_for_pylons.sort(key=lambda a: self.start_location.distance_to(a))
            self.positions_for_buildings.sort(key=lambda a: self.start_location.distance_to(a))

            await self.chat_send("ProLo")
            self.corners = [Point2(Pointlike([0, self.game_info.map_size[1]])),
                            Point2(Pointlike([0, 0])),
                            Point2(Pointlike([self.game_info.map_size[0], self.game_info.map_size[1]])),
                            Point2(Pointlike([self.game_info.map_size[0], 0]))]
            closest_dist = self.units(UnitTypeId.NEXUS).first.distance_to(self.corners[0])
            closest = 0
            for corner in range(1, len(self.corners)):
                corner_dist = self.units(UnitTypeId.NEXUS).first.distance_to(self.corners[corner])
                if corner_dist < closest_dist:
                    closest = corner
                    closest_dist = corner_dist
            del self.corners[closest]

        self.itercount += 1
        # if self.itercount % 10 != 0:
        #     return
        # Get current state (minerals, gas, idles, etc.)
        current_state = sc_helpers.get_player_state(self.state)
        current_state = np.array(current_state)
        my_unit_type_arr = sc_helpers.my_units_to_type_count(self.units)
        enemy_unit_type_arr = sc_helpers.enemy_units_to_type_count(self.known_enemy_units)
        # Get pending
        pending = []
        for unit_type in sc_helpers.MY_POSSIBLES:
            if self.already_pending(unit_type):
                pending.append(1)
            else:
                pending.append(0)

        # Reshape all into batch of 1
        my_unit_type_arr = my_unit_type_arr.reshape(-1)  # batch_size, len
        current_state = current_state.reshape(-1)
        enemy_unit_type_arr = enemy_unit_type_arr.reshape(-1)
        pending = np.array(pending).reshape(-1)
        last_act = np.array([0]).reshape(-1)
        self.prev_state = np.concatenate((current_state,
                                          my_unit_type_arr,
                                          enemy_unit_type_arr,
                                          pending,
                                          last_act))

        action = self.agent.get_action(self.prev_state)
        self.last_reward = await self.activate_sub_bot(action)
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
        elif general_bot_out == 42:
            if SUPER_DEBUG:
                print("Scouting")
            try:
                return await self.send_scout()
            except Exception as e:
                print("Scouting Exception", e)
                return FAILED_REWARD
        elif general_bot_out == 41:
            if SUPER_DEBUG:
                print("Defending")
            try:
                return await self.defend()
            except Exception as e:
                print("Defending exception", e)
                return FAILED_REWARD
        elif general_bot_out == 40:
            if SUPER_DEBUG:
                print("Mining")
            try:
                # return await self.back_to_mining()
                if self.prev_state[4] > 0 or self.prev_state[9] > self.prev_state[28]*20:
                    await self.distribute_workers()
                    return SUCCESS_MINING_REWARD
                else:
                    return FAILED_REWARD
                # return self.mining_reward
            except Exception as e:
                print("Mining Exception", e)
                return FAILED_REWARD
            # return await self.distribute_workers()
        elif general_bot_out == 39 or general_bot_out == 13:
            if SUPER_DEBUG:
                print("Attacking")
            # return await self.activate_combat()
            try:
                return await self.army_attack()
            except Exception as e:
                print("Attacking Exception", e)
                return FAILED_REWARD
        else:
            if SUPER_DEBUG:
                print("Building")
            try:
                return await self.activate_builder(general_bot_out)
            except Exception as e:
                print("Builder Exception", e)
                return FAILED_REWARD

    async def activate_builder(self, hard_coded_choice):
        """
        call on the builder_bot to choose a unit to create, and select a target location if applicable
        :param hard_coded_choice: output from the RL agent
        :return: reward for action based on legality
        """
        unit_choice = hard_coded_choice
        if SUPER_DEBUG:
            print(unit_choice)

        # Need to know unit type to know if placement is valid.
        success = FAILED_REWARD
        index_to_unit = {
            0: UnitTypeId.NEXUS,
            1: UnitTypeId.PYLON,
            2: UnitTypeId.ASSIMILATOR,
            3: UnitTypeId.GATEWAY,
            4: UnitTypeId.WARPGATE,
            5: UnitTypeId.FORGE,
            6: UnitTypeId.CYBERNETICSCORE,
            7: UnitTypeId.PHOTONCANNON,
            8: UnitTypeId.SHIELDBATTERY,
            9: UnitTypeId.ROBOTICSFACILITY,
            10: UnitTypeId.STARGATE,
            11: UnitTypeId.TWILIGHTCOUNCIL,
            12: UnitTypeId.ROBOTICSBAY,
            13: UnitTypeId.FLEETBEACON,
            14: UnitTypeId.TEMPLARARCHIVE,
            15: UnitTypeId.DARKSHRINE
        }
        if unit_choice < 16:
            # If the builder wants to build a building, we need to place it:
            if unit_choice == 0:
                # Nexuses will automatically move into expansion slots
                target_pt = None
            elif unit_choice == 2:
                # Assimilators will automatically move onto vespene geysers
                target_pt = None
            elif unit_choice == 4:
                # Warpgates are upgrades of existing Gateways
                target_pt = None
            elif unit_choice == 1:
                target_pt = self.positions_for_pylons[0]
                pos_ind = 0
            else:
                target_pt = self.positions_for_buildings[0]
                pos_ind = 0
            if target_pt is None:
                try:
                    success = await self.build_building(unit_choice, None)
                except Exception as e:
                    if DEBUG:
                        print("exception", e)
                    success = FAILED_REWARD
            elif unit_choice == 1 or self.state.psionic_matrix.covers(target_pt):
                while True:
                    if unit_choice == 1 and await self.can_place(index_to_unit[unit_choice], target_pt):
                        break  # TODO: Eventually I'd like to pop(pos_ind) off of the list...
                    elif self.state.psionic_matrix.covers(target_pt) and await self.can_place(index_to_unit[unit_choice], target_pt):
                        break  # TODO: Eventually I'd like to pop(pos_ind) off of the list...
                    else:
                        pos_ind += 1
                        if unit_choice == 1:
                            if pos_ind >= len(self.positions_for_pylons):
                                target_pt = await self.find_placement(index_to_unit[unit_choice], self.units(UnitTypeId.NEXUS).random.position)
                                break
                            target_pt = self.positions_for_pylons[pos_ind]
                        else:
                            if pos_ind >= len(self.positions_for_buildings):
                                target_pt = await self.find_placement(index_to_unit[unit_choice], self.units(UnitTypeId.PYLON).random.position)
                                break
                            target_pt = self.positions_for_buildings[pos_ind]
                try:
                    # If can place and psionic covers (or is pylon)
                    success = await self.build_building(unit_choice, target_pt)
                except Exception as e:
                    if DEBUG:
                        print("Exception", e)
                    success = FAILED_REWARD
            else:
                success = FAILED_REWARD
        elif unit_choice < 34:
            try:
                success = await self.train_unit(unit_choice)
            except Exception as e:
                if DEBUG:
                    print("Exception", e)
                success = FAILED_REWARD
        elif unit_choice < 39:
            try:
                success = await self.research_upgrade(unit_choice)
            except Exception as e:
                if DEBUG:
                    print("Exception", e)
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
            0: UnitTypeId.NEXUS,
            1: UnitTypeId.PYLON,
            2: UnitTypeId.ASSIMILATOR,
            3: UnitTypeId.GATEWAY,
            4: UnitTypeId.WARPGATE,
            5: UnitTypeId.FORGE,
            6: UnitTypeId.CYBERNETICSCORE,
            7: UnitTypeId.PHOTONCANNON,
            8: UnitTypeId.SHIELDBATTERY,
            9: UnitTypeId.ROBOTICSFACILITY,
            10: UnitTypeId.STARGATE,
            11: UnitTypeId.TWILIGHTCOUNCIL,
            12: UnitTypeId.ROBOTICSBAY,
            13: UnitTypeId.FLEETBEACON,
            14: UnitTypeId.TEMPLARARCHIVE,
            15: UnitTypeId.DARKSHRINE,
            # 16: UnitTypeId.ORACLESTASISTRAP
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
            if building_target == UnitTypeId.ASSIMILATOR:
                nexi = self.units(UnitTypeId.NEXUS).ready
                for nexus in nexi:
                    geysers_nearby = self.state.vespene_geyser.closer_than(20.0, nexus)

                    for geyser in geysers_nearby:
                        worker = self.select_build_worker(geyser.position, force=True)
                        if not self.units(UnitTypeId.ASSIMILATOR).closer_than(1.0, geyser).exists and worker is not None:
                            self.action_buffer.append(worker.build(UnitTypeId.ASSIMILATOR, geyser))
                            return SUCCESS_BUILD_REWARD
                return FAILED_REWARD
            elif building_target == UnitTypeId.WARPGATE:
                for gateway in self.units(UnitTypeId.GATEWAY).ready:
                    abilities = await self.get_available_abilities(gateway)
                    if AbilityId.MORPH_WARPGATE in abilities:
                        if self.can_afford(AbilityId.MORPH_WARPGATE):
                            self.action_buffer.append(gateway(AbilityId.MORPH_WARPGATE))
                            return SUCCESS_BUILD_REWARD
                    else:
                        if self.units(UnitTypeId.CYBERNETICSCORE).exists:
                            core = self.units(UnitTypeId.CYBERNETICSCORE).random
                            abilities = await self.get_available_abilities(core)
                            if AbilityId.RESEARCH_WARPGATE in abilities:
                                    self.action_buffer.append(core(AbilityId.RESEARCH_WARPGATE))
                                    return SUCCESS_BUILD_REWARD
                return FAILED_REWARD
            elif building_target == UnitTypeId.NEXUS:
                location = await self.get_next_expansion()
                location = await self.find_placement(UnitTypeId.NEXUS, location, placement_step=1)
                worker = self.select_build_worker(location, force=True)
                if worker is not None:
                    self.action_buffer.append(worker.build(UnitTypeId.NEXUS, location))
                    if self.units(NEXUS).amount > 4:
                        return FAILED_REWARD
                    return SUCCESS_BUILD_REWARD
                else:
                    return FAILED_REWARD
            else:
                # pos_dist = random.random()*2 + 3
                pos_dist = 0.5
                pos = placement_location.position.to2.towards(random.choice(self.corners), pos_dist)

                worker = self.select_build_worker(pos, force=True)
                if worker is not None:
                    self.action_buffer.append(worker.build(building_target, pos))
                    # await self.build(building_target, near=pos)
                    if building_target == UnitTypeId.PYLON:
                        return SUCCESS_BUILD_REWARD*0.1
                    if self.units(building_target).amount > 3:
                        return FAILED_REWARD
                    return SUCCESS_BUILD_REWARD
                else:
                    return FAILED_REWARD
        return FAILED_REWARD

    async def train_unit(self, action_unit_index):
        action_to_unit = {
            16: UnitTypeId.PROBE,
            17: UnitTypeId.ZEALOT,
            18: UnitTypeId.STALKER,
            19: UnitTypeId.SENTRY,
            20: UnitTypeId.ADEPT,
            21: UnitTypeId.HIGHTEMPLAR,
            22: UnitTypeId.DARKTEMPLAR,
            23: UnitTypeId.OBSERVER,
            24: UnitTypeId.WARPPRISM,
            25: UnitTypeId.IMMORTAL,
            26: UnitTypeId.COLOSSUS,
            27: UnitTypeId.DISRUPTOR,
            28: UnitTypeId.PHOENIX,
            29: UnitTypeId.VOIDRAY,
            30: UnitTypeId.ORACLE,
            31: UnitTypeId.TEMPEST,
            32: UnitTypeId.CARRIER,
            33: UnitTypeId.MOTHERSHIP,
            # 34: UnitTypeId.INTERCEPTOR,  # TRAIN BY DEFAULT, DONT NEED TO TRAIN
            # 35: UnitTypeId.ARCHON,  # CURRENT IMPOSSIBLE WITH THE API
        }
        unit_to_build = action_to_unit[action_unit_index]
        if DEBUG:
            print("Training a", unit_to_build.name)
        warpgate_gateway_units = [UnitTypeId.ZEALOT, UnitTypeId.STALKER, UnitTypeId.ADEPT, UnitTypeId.HIGHTEMPLAR,
                                  UnitTypeId.DARKTEMPLAR,  UnitTypeId.SENTRY]
        robotics_units = [UnitTypeId.OBSERVER, UnitTypeId.IMMORTAL, UnitTypeId.WARPPRISM, UnitTypeId.COLOSSUS,
                          UnitTypeId.DISRUPTOR]
        stargate_units = [UnitTypeId.PHOENIX, UnitTypeId.ORACLE, UnitTypeId.VOIDRAY, UnitTypeId.TEMPEST,
                          UnitTypeId.CARRIER]
        if unit_to_build == UnitTypeId.PROBE:
            nexus = self.units(UnitTypeId.NEXUS).ready.random
            if nexus.noqueue and self.can_afford(unit_to_build):
                self.action_buffer.append(nexus.train(unit_to_build))
                return SUCCESS_TRAIN_REWARD
            else:
                return FAILED_REWARD
        elif unit_to_build in warpgate_gateway_units:
            # Look for a warpgate first, since those are apparently way better
                if self.units(UnitTypeId.WARPGATE).ready.exists:
                    if self.units(UnitTypeId.PYLON).ready.exists:
                        warp_at = self.units(UnitTypeId.PYLON).ready.random
                    else:
                        warp_at = self.units(UnitTypeId.NEXUS).ready.random
                    abilities_necessary = [AbilityId.WARPGATETRAIN_ZEALOT, AbilityId.WARPGATETRAIN_STALKER,
                                           AbilityId.TRAINWARP_ADEPT, AbilityId.WARPGATETRAIN_HIGHTEMPLAR,
                                           AbilityId.WARPGATETRAIN_DARKTEMPLAR, AbilityId.WARPGATETRAIN_SENTRY]
                    ability_req = abilities_necessary[warpgate_gateway_units.index(unit_to_build)]
                    building = self.units(UnitTypeId.WARPGATE).ready.random
                    abilities = await self.get_available_abilities(building)
                    # all the units have the same cooldown anyway so let's just look at ZEALOT
                    if ability_req in abilities and self.can_afford(unit_to_build):
                        pos = warp_at.position.to2.random_on_distance(4)
                        placement = await self.find_placement(ability_req, pos, placement_step=1)
                        if placement is None:
                            # Can't place
                            return FAILED_REWARD
                        self.action_buffer.append(building.warp_in(unit_to_build, placement))
                        return SUCCESS_TRAIN_REWARD
                    else:
                        return FAILED_REWARD
                elif self.units(UnitTypeId.GATEWAY).ready.exists:
                    building = self.units(UnitTypeId.GATEWAY).ready.random
                    abilities_necessary = [AbilityId.GATEWAYTRAIN_ZEALOT, AbilityId.GATEWAYTRAIN_STALKER,
                                           AbilityId.TRAIN_ADEPT, AbilityId.GATEWAYTRAIN_HIGHTEMPLAR,
                                           AbilityId.GATEWAYTRAIN_DARKTEMPLAR, AbilityId.GATEWAYTRAIN_SENTRY]
                    ability_req = abilities_necessary[warpgate_gateway_units.index(unit_to_build)]
                    abilities = await self.get_available_abilities(building)
                    if ability_req in abilities and self.can_afford(unit_to_build):
                        self.action_buffer.append(building.train(unit_to_build))
                        return SUCCESS_TRAIN_REWARD*3
                    else:
                        # Can't afford or can't make unit type
                        return FAILED_REWARD
        elif unit_to_build in robotics_units:
            if self.units(UnitTypeId.ROBOTICSFACILITY).ready.exists:

                building = self.units(UnitTypeId.ROBOTICSFACILITY).ready.random
                abilities_necessary = [AbilityId.ROBOTICSFACILITYTRAIN_OBSERVER,
                                       AbilityId.ROBOTICSFACILITYTRAIN_IMMORTAL,
                                       AbilityId.ROBOTICSFACILITYTRAIN_WARPPRISM,
                                       AbilityId.ROBOTICSFACILITYTRAIN_COLOSSUS, AbilityId.TRAIN_DISRUPTOR]
                ability_req = abilities_necessary[robotics_units.index(unit_to_build)]
                abilities = await self.get_available_abilities(building)
                if ability_req in abilities and self.can_afford(unit_to_build):
                    self.action_buffer.append(building.train(unit_to_build))
                    return SUCCESS_TRAIN_REWARD
                else:
                    return FAILED_REWARD
            else:
                return FAILED_REWARD
        elif unit_to_build in stargate_units:
            if self.units(UnitTypeId.STARGATE).ready.exists:
                building = self.units(UnitTypeId.STARGATE).ready.random
                abilities_necessary = [AbilityId.STARGATETRAIN_PHOENIX, AbilityId.STARGATETRAIN_ORACLE,
                                       AbilityId.STARGATETRAIN_VOIDRAY,
                                       AbilityId.STARGATETRAIN_CARRIER, AbilityId.STARGATETRAIN_TEMPEST]
                ability_req = abilities_necessary[stargate_units.index(unit_to_build)]
                abilities = await self.get_available_abilities(building)
                if ability_req in abilities and self.can_afford(unit_to_build):
                    self.action_buffer.append(building.train(unit_to_build))
                    return SUCCESS_TRAIN_REWARD*5
                else:
                    return FAILED_REWARD
            else:
                return FAILED_REWARD
        elif unit_to_build == UnitTypeId.MOTHERSHIP:
            nexus = self.units(UnitTypeId.NEXUS).ready.random
            nexus_abilities = await self.get_available_abilities(nexus)
            if self.can_afford(unit_to_build) and AbilityId.NEXUSTRAINMOTHERSHIP_MOTHERSHIP in nexus_abilities:
                self.action_buffer.append(nexus.train(unit_to_build))
                return SUCCESS_TRAIN_REWARD
            else:
                return FAILED_REWARD
        return FAILED_REWARD

    async def army_attack(self):
        """
        :return: reward based on if we sent out attackers
        """
        def in_range(unit, target, gap=0.5):
            # if gap > 0, in_range will be true if unit is closer than maximum fire range
            return unit.distance_to(target) + gap < unit.radius + target.radius + max(unit.air_range, unit.ground_range)

        def half_health(unit):
            return unit.health + unit.shield < min(unit.health_max, unit.shield_max)

        attacking_classes = [UnitTypeId.ZEALOT, UnitTypeId.STALKER, UnitTypeId.SENTRY, UnitTypeId.ADEPT,
                             UnitTypeId.HIGHTEMPLAR, UnitTypeId.DARKTEMPLAR, UnitTypeId.IMMORTAL, UnitTypeId.COLOSSUS,
                             UnitTypeId.PHOENIX, UnitTypeId.VOIDRAY, UnitTypeId.ORACLE, UnitTypeId.TEMPEST,
                             UnitTypeId.CARRIER, UnitTypeId.INTERCEPTOR, UnitTypeId.MOTHERSHIP]

        all_targets = self.known_enemy_units.exclude_type([UnitTypeId.LARVA, UnitTypeId.EGG])
        ground_targets = all_targets.not_flying
        flying_targets = all_targets.flying
        attacking_ground_targets = ground_targets.filter(lambda x: (x.can_attack_ground or x.can_attack_air))
        attacking_flying_targets = flying_targets.filter(lambda x: (x.can_attack_ground or x.can_attack_air))
        my_army = self.units.of_type(attacking_classes)
        army_half_health = len([unit for unit in my_army if half_health(unit)])
        # if army_half_health <= self.army_below_half and self.state.game_loop - self.last_attack_loop > 120:  # If nobody got hurt and we did this recently, save time and don't do it again.
        #     self.last_attack_loop = self.state.game_loop
        #     self.army_below_half = army_half_health
        #     return SUCCESS_ATTACK_REWARD
        self.last_attack_loop = self.state.game_loop
        self.army_below_half = army_half_health

        for unit in my_army:
            if unit.type_id == UnitTypeId.SENTRY and await self.can_cast(unit, AbilityId.GUARDIANSHIELD_GUARDIANSHIELD):
                await self.do(unit(AbilityId.GUARDIANSHIELD_GUARDIANSHIELD))

            target = None
            nearest_target = None
            possible_targets = all_targets.filter(lambda x: unit.target_in_range(x))
            if unit.type_id == UnitTypeId.STALKER and attacking_flying_targets.exists:
                target = attacking_flying_targets.closest_to(unit)
            elif possible_targets:
                    nearest_target = possible_targets.closest_to(unit)
                    target = min(possible_targets, key=lambda t: (t.health + t.shield) /
                                                                 (t.ground_dps + t.air_dps + 1))
            treat_outrange = -100
            if all_targets:
                if unit.type_id is UnitTypeId.COLOSSUS:
                    treat = max(all_targets, key=lambda e: max(e.ground_range, e.air_range)** 2 -
                                                                e.position._distance_squared(unit.position))
                    treat_outrange = unit.radius + treat.radius + max(treat.ground_range, treat.air_range) + 1 - \
                                     unit.distance_to(treat)
                elif unit.is_flying:
                    treat = max(all_targets, key=lambda e: e.air_range ** 2 -
                                                                e.position._distance_squared(unit.position))
                    treat_outrange = unit.radius + treat.radius + treat.air_range + 1 - \
                                     unit.distance_to(treat)
                else:
                    treat = max(all_targets, key=lambda e: e.ground_range ** 2 -
                                                                e.position._distance_squared(unit.position))
                    treat_outrange = unit.radius + treat.radius + treat.ground_range + 1 - \
                                     unit.distance_to(treat)

            if target is None:
                if attacking_flying_targets.exists and unit.can_attack_air:
                    target = attacking_flying_targets.closest_to(unit)
                elif attacking_ground_targets.exists:
                    target = attacking_ground_targets.closest_to(unit)
                elif flying_targets.exists and unit.can_attack_air:
                    target = flying_targets.closest_to(unit)
                elif ground_targets.exists:
                    target = ground_targets.closest_to(unit)
                else:
                    target = random.choice(self.enemy_start_locations).position
            retreat = False
            if unit.type_id is not UnitTypeId.ZEALOT:
                if (unit.type_id is not UnitTypeId.VOIDRAY and unit.weapon_cooldown > 2 and
                    possible_targets.filter(lambda x: (x.can_attack_air or x.can_attack_ground)).exists
                    and nearest_target is not None and (in_range(unit, nearest_target) or treat_outrange >= 0)) or \
                        (army_half_health < my_army.amount / 2 and half_health(unit) and treat_outrange+unit.radius*2>=0
                        or unit.type_id is UnitTypeId.SENTRY):
                    # target = self.start_location
                    retreat = True
                    if unit.distance_to(self.start_location) < 5:
                        target = all_targets.closest_to(unit)
            unit_abs = await self.get_available_abilities(unit)
            if AbilityId.EFFECT_BLINK_STALKER in unit_abs and await self.can_cast(unit, AbilityId.EFFECT_BLINK_STALKER)\
                    and unit.distance_to(target) > 6 and \
                    target is not UnitTypeId.ZERGLING:  # don't jump into zerglings
                if not retreat:
                    towards = -unit.radius + unit.ground_range  # one corpus closer
                    if isinstance(target, sc2Unit):
                        towards += target.radius
                        target = target.position.towards(unit.position, towards)
                    else:
                        target = target.towards(unit.position, towards)
                await self.do(unit(AbilityId.EFFECT_BLINK_STALKER, target))
                if retreat and possible_targets.exists:
                    # blink in other direction if can't blink back to base
                    alt_target = unit.position.towards(possible_targets.center, -8)
                    self.action_buffer.append(unit(AbilityId.EFFECT_BLINK_STALKER, alt_target, queue=True))
            if retreat:
                self.action_buffer.append(unit.move(target))
            else:
                self.action_buffer.append(unit.attack(target))
        attack_reward = min((len(my_army)-5)*SUCCESS_ATTACK_REWARD, 10)
        return attack_reward

    async def defend(self):
        """
        This function find enemy army centroids, and the number of attackers in each army
        so if we're being hit on 3 fronts, it'll attempt to distribute to each.
        #TODO: intelligently grab first available attackers (like.. grab nearest)
        :return: success for a defensive mobilization, fail for nobody to defend or nothing to defend
        """
        enemy_positions = []
        num_attackers = []
        for depo in self.units(UnitTypeId.NEXUS):
            num_attackers.append(0)
            have_position = False
            for unit in self.known_enemy_units.not_structure:
                if unit.position.to2.distance_to(depo.position.to2) < 15:
                    if not have_position:
                        enemy_positions.append(unit.position)
                        have_position = True
                    num_attackers[-1] += 1
        defensive_orders = []
        for position, army_size in zip(enemy_positions, num_attackers):
            attacking_classes = [UnitTypeId.ZEALOT, UnitTypeId.STALKER, UnitTypeId.SENTRY, UnitTypeId.ADEPT,
                                 UnitTypeId.HIGHTEMPLAR, UnitTypeId.DARKTEMPLAR, UnitTypeId.IMMORTAL,
                                 UnitTypeId.COLOSSUS,
                                 UnitTypeId.PHOENIX, UnitTypeId.VOIDRAY, UnitTypeId.ORACLE, UnitTypeId.TEMPEST,
                                 UnitTypeId.CARRIER, UnitTypeId.INTERCEPTOR, UnitTypeId.MOTHERSHIP]
            for unit_type in attacking_classes:
                for unit in self.units(unit_type).idle:
                    defensive_orders.append(unit.attack(position))
                    if len(defensive_orders) > army_size:
                        break
            if len(defensive_orders) < army_size:
                for worker in self.workers:
                    defensive_orders.append(worker.attack(position))
                    if len(defensive_orders) > army_size:
                        break

        self.action_buffer.extend(defensive_orders)
        if len(defensive_orders) > 0:
            return SUCCESS_ATTACK_REWARD
        else:
            return FAILED_REWARD

    async def send_scout(self):
        """
        :return: fail if we scouted too recently, success otherwise
        """
        # TODO: add global list of recently checked base locations so that I can prioritize based on recency of check
        # TODO: use global list of recency / discovery to reward bot for scouting that pays off
        # If I have scouted recently (within 200 iterations?), don't re-scout
        if self.itercount - self.last_scout_iteration < 200:
            return FAILED_REWARD
        a_scout = None
        places_to_look = []
        scout_loc = random.choice(self.enemy_start_locations)

        for base_location in self.expansion_locations.keys():
            if base_location.position in self.scouted_bases:
                continue
            else:
                places_to_look.append(base_location)
        if len(places_to_look) == 0:
            self.scouted_bases = []

        if len(self.known_enemy_structures) < 1:
            # If we haven't found their main yet, prioritize enemy base locs
            enemy_base_locs = self.enemy_start_locations
            for enemy_base_loc in enemy_base_locs:
                invalid = False
                # This one is a candidate
                if enemy_base_loc.position in self.scouted_bases:
                    invalid = True
                    # If we've been nearby, it's no longer valid
                if not invalid:
                    # If we haven't been nearby, this is valid. break search
                    scout_loc = enemy_base_loc
                    break
        elif len(places_to_look) > 0:
            scout_loc = random.choice(places_to_look).position
        else:
            scout_loc = random.choice(self.enemy_start_locations)

        if self.units(UnitTypeId.OBSERVER).exists:
            a_scout = self.units(UnitTypeId.OBSERVER).closest_to(scout_loc)
        elif self.workers.exists:
            if self.workers.idle.exists:
                a_scout = self.workers.idle.closest_to(scout_loc)
            else:
                a_scout = self.workers.closest_to(scout_loc)

        if a_scout is not None:
            self.action_buffer.append(a_scout.move(scout_loc))
            self.last_scout_iteration = self.itercount  # Track when we sent out our last scout
            self.scouted_bases.append(scout_loc)
            return SUCCESS_SCOUT_REWARD
        else:
            return FAILED_REWARD

    async def research_upgrade(self, research_index):
        index_to_upgrade = {
            34: "ground_attacks",
            35: "air_attacks",
            36: "ground_armor",
            37: "air_armor",
            38: "shields",
            # 39: "speed",
            # 40: "range",
            # 41: "spells",
            # 42: "misc"
        }
        research_topic = index_to_upgrade[research_index]
        if research_topic in ["ground_attacks", "ground_armor", "shields"]:
            if self.units(UnitTypeId.FORGE).exists:
                forge = self.units(UnitTypeId.FORGE).random
                abilities = await self.get_available_abilities(forge)
                if research_topic == "ground_attacks":
                    all_topics = [AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL1,
                                  AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL2,
                                  AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL3]
                elif research_topic == "ground_armor":
                    all_topics = [AbilityId.FORGERESEARCH_PROTOSSGROUNDARMORLEVEL1,
                                  AbilityId.FORGERESEARCH_PROTOSSGROUNDARMORLEVEL2,
                                  AbilityId.FORGERESEARCH_PROTOSSGROUNDARMORLEVEL3]
                elif research_topic == "shields":
                    all_topics = [AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL1,
                                  AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL2,
                                  AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL3]
                for topic in all_topics:
                    if topic in abilities:
                        self.action_buffer.append(forge(topic))
                        return SUCCESS_BUILD_REWARD
            return FAILED_REWARD
        elif research_topic in ["air_attacks", "air_armor"]:
            if self.units(UnitTypeId.CYBERNETICSCORE).exists:
                core = self.units(UnitTypeId.CYBERNETICSCORE).random
                abilities = await self.get_available_abilities(core)
                if research_topic == "air_attacks":
                    air_research = [AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL1,
                                    AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL2,
                                    AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL3]
                elif research_topic == "air_armor":
                    air_research = [AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL1,
                                    AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL1,
                                    AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL1]
                for topic in air_research:
                    if topic in abilities:
                        self.action_buffer.append(core(topic))
                        return SUCCESS_BUILD_REWARD
            return FAILED_REWARD

    async def back_to_mining(self):
        reassigned_miners = 0
        for a in self.units(UnitTypeId.ASSIMILATOR):
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
        print("Game over!")
        if game_result == sc2.Result.Defeat:
            reward = -250  # + self.itercount/500.0 + self.units.amount
        elif game_result == sc2.Result.Tie:
            reward = 50
        elif game_result == sc2.Result.Victory:
            reward = 250  # - min(self.itercount/500.0, 900) + self.units.amount
        else:
            # ???
            return -13
        bot_fn = '../txts/' + self.agent.bot_name + '_victories.txt'
        with open(bot_fn, "a") as myfile:
            myfile.write(str(reward) + '\n')
        self.agent.save_reward(reward)
        R = 0
        rewards = []
        all_rewards = self.agent.replay_buffer.rewards_list
        reward_sum = sum(all_rewards)
        all_values = self.agent.replay_buffer.value_list
        deeper_all_values = self.agent.replay_buffer.deeper_value_list
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
            self.agent.replay_buffer.deeper_advantage_list = deeper_advantages.detach().clone().cpu().numpy().tolist()
        else:
            self.agent.replay_buffer.deeper_advantage_list = [None] * len(all_rewards)
        # Scale rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + torch.Tensor([np.finfo(np.float32).eps]))
        advantages = (advantages - advantages.mean()) / (advantages.std() + torch.Tensor([np.finfo(np.float32).eps]))
        self.agent.replay_buffer.rewards_list = rewards.detach().clone().cpu().numpy().tolist()
        self.agent.replay_buffer.advantage_list = advantages.detach().clone().cpu().numpy().tolist()
        return reward_sum


def run_episode(q, main_agent):
    result = None
    agent_in = main_agent.duplicate()

    bot = StarmniBot(rl_agent=agent_in)
    opponents = [
        Computer(Race.Zerg, Difficulty.Medium),
        # Computer(Race.Protoss, Difficulty.VeryEasy),
        # Computer(Race.Terran, Difficulty.VeryEasy)
    ]
    enemy = random.choice(opponents)

    try:
        result = sc2.run_game(sc2.maps.get("Acid Plant LE"),
                              [Bot(Race.Protoss, bot), enemy],
                              realtime=False)
    except KeyboardInterrupt:
        result = [-1, -1]
    except Exception as e:
        print(str(e))
        print("No worries", e, " carry on please")
    if type(result) == list and len(result) > 1:
        result = result[0]

    reward_sum = bot.finish_episode(result)
    if q is not None:
        try:
            q.put([reward_sum, bot.agent.replay_buffer.__getstate__()])
        except RuntimeError as e:
            print(e)
            return [reward_sum, bot.agent.replay_buffer.__getstate__()]
    return [reward_sum, bot.agent.replay_buffer.__getstate__()]


# def main(episodes, agent_in, num_processes, reset_on_fail=False):
#     running_reward_array = []
#     # lowered = False
#     agent = agent_in.duplicate()
# #     mp.set_start_method('spawn')
#     for episode in range(episodes):
#         successful_runs = 0
#         master_reward, reward, running_reward = 0, 0, 0
#         processes = []
#         queueue = mp.Manager().Queue()
#         for proc in range(num_processes):
#             p = mp.Process(target=run_episode, args=(queueue, agent))
#             p.start()
#             processes.append(p)
#         for p in processes:
#             p.join()
#         while not queueue.empty():
#             try:
#                 fake_out = queueue.get()
#             except MemoryError as e:
#                 print(e)
#                 fake_out = [-13, None]
#             if fake_out[0] != -13:
#                 master_reward += fake_out[0]
#                 running_reward_array.append(fake_out[0])
#                 agent.replay_buffer.extend(fake_out[1])
#                 successful_runs += 1
#
#         if successful_runs > 0:
#             if (reset_on_fail and master_reward > 0) or not reset_on_fail:
#                 reward = master_reward / float(successful_runs)
#                 agent.end_episode(reward, num_processes)
#                 running_reward = sum(running_reward_array[-100:]) / float(min(100.0, len(running_reward_array)))
#                 agent.save('../models/last')
#             else:
#                 agent = agent_in.duplicate()
#                 print("Resetting...")
#
#         if episode % 50 == 0:
#             print(f'Episode {episode}  Last Reward: {reward}  Average Reward: {running_reward}')
#             # print(f"Running {num_processes} concurrent simulations per episode")
#     return running_reward_array


def bernoulli_main(episodes, agent_in, num_processes):
    def bernoulli_test(p, n, k, alpha):
        coef = math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
        p_to_k = p ** k
        q_to_n_k = (1 - p) ** (n - k)
        test_res = coef * p_to_k * q_to_n_k
        if test_res < alpha:
            return True
        else:
            return False

    win_prob = 0.25
    min_games = 15
    alpha = 0.05
    k, n, successful_runs, master_reward, reward, running_reward = 0, 0, 0, 0, 0, 0
    find_new_step = True
    running_reward_array = []
    # lowered = False
    agent = agent_in.duplicate()
    # mp.set_start_method('spawn')
    for episode in range(6, episodes):
        try:
            last_agent = agent.duplicate()
            if find_new_step:
                successful_runs = 0
                master_reward, reward, running_reward = 0, 0, 0
                tuple_out = run_episode(None,
                                        agent)
                if tuple_out[0] != -13:
                    master_reward += tuple_out[0]
                    running_reward_array.append(tuple_out[0])
                    agent.replay_buffer.extend(tuple_out[1])
                    successful_runs += 1
            if successful_runs > 0 or not find_new_step:  # if we have a non-empty replay buffer
                reward = master_reward
                agent.end_episode(reward, num_processes)  # take a gradient step
                running_reward = sum(running_reward_array[-100:]) / float(min(100.0, len(running_reward_array)))
                while True:  # Do the bernoulli testing

                    tuple_out = run_episode(None, agent)
                    if tuple_out[0] != -13:
                        if tuple_out[1]['rewards'][-1] > 0:
                            k += 1
                        n += 1
                        if n < num_processes:
                            master_reward += tuple_out[0]
                            agent.replay_buffer.extend(tuple_out[1])
                    if n >= min_games:
                        new_win_prob = float(k)/float(n)
                        if bernoulli_test(p=win_prob, n=n, k=k, alpha=alpha):
                            if new_win_prob > win_prob:
                                win_prob = new_win_prob
                                k = 0
                                n = 0
                                find_new_step = False
                                agent.save(f'../models/{episode}')
                                break
                            else:
                                agent = last_agent
                                k = 0
                                n = 0
                                find_new_step = True
                                break
                    print(f"After this episode, n={n}")
                    if n > 100 and k > n*.95:
                        print("Finishing this model")
                        agent.save('FINAL')
                        return

            if episode % 50 == 0:
                print(f'Episode {episode}  Last Reward: {reward}  Average Reward: {running_reward}')
                # print(f"Running {num_processes} concurrent simulations per episode")
        except RuntimeError as e:
            print(e)
            find_new_step = True
            time.sleep(5)
            continue
    return running_reward_array


if __name__ == '__main__':
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
    # torch.set_num_threads(NUM_PROCS)
    dim_in = 194
    dim_out = 44
    bot_name = AGENT_TYPE + 'SC_Macro'+'Medium'
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
                             sl_init=SL_INIT)
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
    start_time = time.time()

    # main(episodes=NUM_EPS, agent_in=policy_agent, num_processes=NUM_PROCS, reset_on_fail=True)
    policy_agent.load('../models/5')
    bernoulli_main(episodes=NUM_EPS, agent_in=policy_agent, num_processes=NUM_PROCS)
    with open(bot_name+'time.txt', 'a') as writer:
        writer.write(str(time.time()-start_time))
