from sc2.constants import *
from sc2.ids.unit_typeid import *
import numpy as np
from sc2 import Race
from enum import Enum

class EXPANDED_TYPES(Enum):
    ARMY_COUNT = 0
    FOOD_ARMY = 1
    FOOD_CAP = 2
    FOOD_USED = 3
    IDLE_WORKER_COUNT = 4
    LARVA_COUNT = 5
    MINERALS = 6
    VESPENE = 7
    WARP_GATE_COUNT = 8
    COMMAND_CENTER = 9
    SUPPLY_DEPOT = 10
    REFINERY = 11
    BARRACKS = 12
    ENGINEERING_BAY = 13
    ARMORY = 14
    FACTORY = 15
    STARPORT = 16
    SCV = 17
    MARINE = 18
    PENDING_COMMAND_CENTER = 19
    PENDING_SUPPLY_DEPOT = 20
    PENDING_REFINERY = 21
    PENDING_BARRACKS = 22
    PENDING_ENGINEERING_BAY = 23
    PENDING_ARMORY = 24
    PENDING_FACTORY = 25
    PENDING_STARPORT = 26
    PENDING_SCV = 27
    PENDING_MARINE = 28
    PENDING_HELLION = 29
    HELLION = 30
    LAST_ACTION = 31

MY_POSSIBLES = [COMMANDCENTER,
                SUPPLYDEPOT,
                REFINERY,
                BARRACKS,
                ENGINEERINGBAY,
                ARMORY,
                FACTORY,
                STARPORT,
                SCV,
                MARINE,
                HELLION]

def my_units_to_str(unit_idx):
    return str(MY_POSSIBLES[unit_idx])

def my_units_to_type_count(unit_array_in):
    """
    Take in current units owned by player and return a 36-dim list of how many of each type there are
    :param unit_array_in: self.units from a python-sc2 bot
    :return: 1x36 where each element is the count of units of that type
    """
    type_counts = np.zeros(len(MY_POSSIBLES))
    for unit in unit_array_in:
        # print(type_counts)
        # print(unit)
        # print(unit.type_id)
        # print(MY_POSSIBLES)
        # print(MY_POSSIBLES.index(unit.type_id))
        type_counts[MY_POSSIBLES.index(unit.type_id)] += 1
    return type_counts


def get_player_state(state_in):

    sorted_observed_player_state = [
        state_in.observation.player_common.army_count,
        state_in.observation.player_common.food_army,
        state_in.observation.player_common.food_cap,
        state_in.observation.player_common.food_used,
        state_in.observation.player_common.idle_worker_count,
        state_in.observation.player_common.larva_count,
        state_in.observation.player_common.minerals,
        state_in.observation.player_common.vespene,
        state_in.observation.player_common.warp_gate_count
    ]
    return sorted_observed_player_state


def get_human_readable_mapping():
    idx_to_name = {}
    name_to_idx = {}

    idx = 0

    # purpose is to make mappings between human-readable names and indices in the statespace:
    # self.prev_state = np.concatenate((current_state,
    #                                   my_unit_type_arr,
    #                                   enemy_unit_type_arr, (not using)
    #                                   pending,
    #                                   last_act))

    # current_state
    current_state_names = [
        'army_count',
        'food_army',
        'food_cap',
        'food_used',
        'idle_worker_count',
        'larva_count',
        'minerals',
        'vespene',
        'warp_gate_count'
    ]

    my_unit_type_names = []
    for e in MY_POSSIBLES:
        e = str(e)
        e = e.replace('UnitTypeId.', '')
        my_unit_type_names.append(e)



    for e in current_state_names + my_unit_type_names:
        e = str(e)
        e = e.replace('UnitTypeId.', '')
        # print(e)
        idx_to_name[idx] = e
        name_to_idx[e] = idx
        idx += 1

    # pending
    for e in my_unit_type_names:
        e = "PENDING_" + e
        idx_to_name[idx] = e
        name_to_idx[e] = idx
        idx += 1

    # last act
    e = 'last_act'
    idx_to_name[idx] = e
    name_to_idx[e] = idx
    idx += 1

    return idx_to_name, name_to_idx

def get_human_readable_action_mapping():
    idx_to_action = {}
    idx = 0
    for e in MY_POSSIBLES:
        e = str(e)
        e = e.replace('UnitTypeId.', '')
        idx_to_action[idx] = e
        idx += 1

    idx_to_action[idx] = 'back_to_mining'
    idx += 1

    return idx_to_action

def get_unit_data(unit_in):
    if unit_in is None:
        return [-1, -1, -1, -1]
    extracted_data = [
        float(unit_in.position.x),
        float(unit_in.position.y),
        float(unit_in.health),
        float(unit_in.weapon_cooldown),
    ]
    return extracted_data


def get_enemy_unit_data(unit_in):
    data = get_unit_data(unit_in)
    if unit_in is None:
        data.append(-1)
    else:
        data.append(float(unit_in.type_id == UnitTypeId.BANELING))
    return data


def get_nearest_enemies(unit_in, enemy_list):
    my_pos = [unit_in.position.x, unit_in.position.y]
    distances = []
    enemies = []
    for enemy in enemy_list:
        enemy_pos = [enemy.position.x, enemy.position.y]
        distances.append(dist(my_pos, enemy_pos))
        enemies.append(enemy)
    sorted_results = [x for _, x in sorted(zip(distances,enemies), key=lambda pair: pair[0])]
    sorted_results.extend([None, None, None, None, None])
    return sorted_results[:5]


def dist(pos1, pos2):
    return np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)
