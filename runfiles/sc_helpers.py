from sc2.constants import *
from sc2.ids.unit_typeid import *
import numpy as np
from sc2 import Race
MY_POSSIBLES = [PROBE, ZEALOT, STALKER, SENTRY, ADEPT, HIGHTEMPLAR, DARKTEMPLAR, OBSERVER, WARPPRISM,
                IMMORTAL, COLOSSUS, DISRUPTOR, PHOENIX, VOIDRAY, ORACLE, TEMPEST, CARRIER, INTERCEPTOR,
                MOTHERSHIP, NEXUS, PYLON, ASSIMILATOR, GATEWAY, WARPGATE, FORGE, CYBERNETICSCORE, PHOTONCANNON,
                SHIELDBATTERY, ROBOTICSFACILITY, STARGATE, TWILIGHTCOUNCIL, ROBOTICSBAY, FLEETBEACON,
                TEMPLARARCHIVE, DARKSHRINE, ORACLESTASISTRAP]

ENEMY_POSSIBLES = [PROBE, ZEALOT, STALKER, SENTRY, ADEPT, HIGHTEMPLAR, DARKTEMPLAR, OBSERVER, ARCHON, WARPPRISM,
     IMMORTAL, COLOSSUS, DISRUPTOR, PHOENIX, VOIDRAY, ORACLE, TEMPEST, CARRIER, INTERCEPTOR,
     MOTHERSHIP, NEXUS, PYLON, ASSIMILATOR, GATEWAY, WARPGATE, FORGE, CYBERNETICSCORE, PHOTONCANNON,
     SHIELDBATTERY, ROBOTICSFACILITY, STARGATE, TWILIGHTCOUNCIL, ROBOTICSBAY, FLEETBEACON,
     TEMPLARARCHIVE, DARKSHRINE, ORACLESTASISTRAP, SCV, MULE, MARINE, REAPER, MARAUDER, GHOST, HELLION, WIDOWMINE,
     CYCLONE, SIEGETANKSIEGED, THOR, VIKINGFIGHTER, MEDIVAC, LIBERATOR, RAVEN, BANSHEE, BATTLECRUISER, COMMANDCENTER,
     SUPPLYDEPOT, REFINERY, ENGINEERINGBAY, BUNKER, MISSILETURRET, SENSORTOWER, GHOSTACADEMY, FACTORY, BARRACKS, STARPORT,
     FUSIONCORE, TECHLAB, REACTOR, AUTOTURRET, ORBITALCOMMAND, PLANETARYFORTRESS, HELLIONTANK, LARVA, DRONE, OVERLORD,
     QUEEN, ZERGLING, BANELING, ROACH, RAVAGER, OVERSEER, CHANGELING, HYDRALISK, LURKER, MUTALISK, CORRUPTOR,
     SWARMHOSTMP, LOCUSTMP, INFESTOR, INFESTEDTERRAN, VIPER, ULTRALISK, BROODLORD, BROODLING, HATCHERY, EXTRACTOR,
     SPAWNINGPOOL, SPINECRAWLER, SPORECRAWLER, EVOLUTIONCHAMBER, ROACHWARREN, BANELINGNEST, HYDRALISKDEN, LURKERDENMP,
     SPIRE, NYDUSNETWORK, INFESTATIONPIT, ULTRALISKCAVERN, CREEPTUMOR, LAIR, HIVE, GREATERSPIRE]
ENEMY_MAPPINGS = {
        WIDOWMINEBURROWED: WIDOWMINE,  # widow mine / burrowed widow mine
        SIEGETANK: SIEGETANKSIEGED,  # siege tank siege / tank
        VIKINGASSAULT: VIKINGFIGHTER,  # viking assault / viking fighter
        COMMANDCENTERFLYING: COMMANDCENTER,  # flying command center / command center
        ORBITALCOMMANDFLYING: ORBITALCOMMAND,  # flying orbital command / command center
        SUPPLYDEPOTLOWERED: SUPPLYDEPOT,  # supply depot / lowered
        BARRACKSFLYING: BARRACKS,  # barracks / flying barracks
        FACTORYFLYING: FACTORY,  # factory / factory flying
        STARPORTFLYING: STARPORT,  # starport / starport flying
        BARRACKSTECHLAB: TECHLAB,  # barracks tech lab / tech lab
        FACTORYTECHLAB: TECHLAB,  # factory tech lab / tech lab
        STARPORTTECHLAB: TECHLAB,  # starport tech lab / tech lab
        BARRACKSREACTOR: REACTOR,  # barracks reactor / reactor
        FACTORYREACTOR: REACTOR,  # factory reactor / reactor
        STARPORTREACTOR: REACTOR,  # starport reactor / reactor
        DRONEBURROWED: DRONE,
        BANELINGBURROWED: BANELING,
        HYDRALISKBURROWED: HYDRALISK,
        ROACHBURROWED: ROACH,
        ZERGLINGBURROWED: ZERGLING,
        QUEENBURROWED: QUEEN,
        RAVAGERBURROWED: RAVAGER,
        CHANGELINGZEALOT: CHANGELING,
        CHANGELINGMARINESHIELD: CHANGELING,
        CHANGELINGMARINE: CHANGELING,
        CHANGELINGZERGLINGWINGS: CHANGELING,
        CHANGELINGZERGLING: CHANGELING,
        LURKERBURROWED: LURKER,
        SWARMHOSTBURROWEDMP: SWARMHOSTMP,
        LOCUSTMPFLYING: LOCUSTMP,
        INFESTORTERRANBURROWED: INFESTOR,
        INFESTORBURROWED: INFESTOR,
        INFESTORTERRAN: INFESTOR,
        ULTRALISKBURROWED: ULTRALISK,
        SPINECRAWLERUPROOTED: SPINECRAWLER,
        SPORECRAWLERUPROOTED: SPORECRAWLER,
        LURKERDEN: LURKERDENMP,
    }

index_to_upgrade = {
            34: "ground_attacks",
            35: "air_attacks",
            36: "ground_armor",
            37: "air_armor",
            38: "shields",
            39: "speed",
            40: "range",
            41: "spells",
            42: "misc"
        }

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

def get_human_readable_mapping():
    idx_to_name = {}
    name_to_idx = {}

    idx = 0

    # purpose is to make mappings between human-readable names and indices in the statespace:
    # self.prev_state = np.concatenate((current_state,
    #                                   my_unit_type_arr,
    #                                   enemy_unit_type_arr,
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

    enemy_unit_type_names = []
    for e in ENEMY_POSSIBLES:
        e = str(e)
        e = e.replace('UnitTypeId.', '')
        enemy_unit_type_names.append(e)

    for e in current_state_names + my_unit_type_names:
        e = str(e)
        e = e.replace('UnitTypeId.', '')
        # print(e)
        idx_to_name[idx] = e
        name_to_idx[e] = idx
        idx += 1

    # for i in index_to_upgrade.keys():
    #     idx_to_name[i] = index_to_upgrade[i]
    #     name_to_idx[index_to_upgrade[i]] = i
    #     idx += 1

    for e in enemy_unit_type_names:
        e = str(e)
        e = e.replace('UnitTypeId.', '')
        # print(e)
        idx_to_name[idx] = e
        name_to_idx[e] = idx
        idx += 1

    # nop
    # idx_to_name[idx] = 'nop'
    # name_to_idx['nop'] = idx
    # idx += 1

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

    # print(idx_to_name)
    # quit()

    return idx_to_name, name_to_idx

def get_human_readable_action_mapping():
    idx_to_action = {}
    idx = 0
    for e in index_to_unit:
        e = index_to_unit[e]
        e = str(e)
        e = e.replace('UnitTypeId.', '')
        idx_to_action[idx] = e
        idx += 1
    for e in action_to_unit:
        e = action_to_unit[e]
        e = str(e)
        e = e.replace('UnitTypeId.', '')
        idx_to_action[idx] = e
        idx += 1
    for i in index_to_upgrade.keys():
        idx_to_action[i] = index_to_upgrade[i]
        idx += 1
    idx_to_action[idx] = 'nop'
    idx += 1
    return idx_to_action

def my_units_to_type_count(unit_array_in):
    """
    Take in current units owned by player and return a 36-dim list of how many of each type there are
    :param unit_array_in: self.units from a python-sc2 bot
    :return: 1x36 where each element is the count of units of that type
    """
    type_counts = np.zeros(len(MY_POSSIBLES))
    for unit in unit_array_in:
        type_counts[MY_POSSIBLES.index(unit.type_id)] += 1
    return type_counts


def enemy_units_to_type_count(enemy_array_in):
    """
    Take in enemy units and map them to type counts, as in my army. But here I consider all 3 races
    :param enemy_array_in: self.known_enemy_units (or my all_enemy_units list) from python-sc2 bot
    :return: 1x111 where each element is the count of units of that type
    """
    type_counts = np.zeros(len(ENEMY_POSSIBLES))
    for unit in enemy_array_in:
        if unit.type_id in ENEMY_POSSIBLES:
            type_counts[ENEMY_POSSIBLES.index(unit.type_id)] += 1
        elif unit.type_id in ENEMY_MAPPINGS.keys():
            type_counts[ENEMY_POSSIBLES.index(ENEMY_MAPPINGS[unit.type_id])] += 1
        else:
            continue
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
