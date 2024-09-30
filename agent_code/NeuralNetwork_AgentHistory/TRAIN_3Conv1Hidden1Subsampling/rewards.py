import copy
from agent_code.TRAIN_3Conv1Hidden1Subsampling.helperFunctions import (action_not_possible, action_leads_to_dying,
                                                                       in_scope_of_bomb_after_action,
                                                                       cannot_escape_after_action,
                                                                       bomb_will_kill_opponent,
                                                                       bomb_might_kill_opponent,
                                                                       bomb_shortens_path_to_coin,
                                                                       bomb_will_destroy_crates,
                                                                       action_leads_to_dying_opponent,
                                                                       collecting_coin, position_after_step,
                                                                       walking_closer_to_reachable_coin,
                                                                       no_coin_reachable, closer_distance_to_coin)


def reward(situationDictionary, action):
    situation = copy.deepcopy(situationDictionary)
    after_action = position_after_step(situation, action)
    ### Impossible action
    if action_not_possible(situation, action, after_action):
        return -8

    ### Action leads to Dying
    if action_leads_to_dying(situation, after_action):
        return -3

    ### Being close to a bomb
    # Walking where a bomb will explode soon:
    bombs = in_scope_of_bomb_after_action(situation, action, after_action)
    if bombs:
        if cannot_escape_after_action(situation, action, after_action, bombs):
            if action == "WAIT":
                return -3
            return -2

    ### Player can survive, try to maximise gain:
    possible_reward = 0
    if action == "BOMB":
        if bomb_will_kill_opponent(situation, after_action):
            possible_reward += 4
        if bomb_might_kill_opponent(situation, after_action):
            possible_reward += 0.2
        if bomb_shortens_path_to_coin(situation, action, after_action):
            possible_reward += 0.25
        if bomb_will_destroy_crates(situation, after_action):
            possible_reward += 0.15
        return possible_reward
    if action != "WAIT":
        possible_reward += 0.1
    else:
        possible_reward -= 0.5
    if action_leads_to_dying_opponent(situation, action, after_action):  # For Example standing in the way
        # and therefore blocking opponent to stand in bomb.
        possible_reward += 4
    if collecting_coin(situation, action, after_action):
        possible_reward += 2
    if walking_closer_to_reachable_coin(situation, after_action):
        possible_reward += 1
    elif not no_coin_reachable(situation, after_action):
        possible_reward -= 1
    if no_coin_reachable(situation, after_action):
        if closer_distance_to_coin(situation, after_action):
            possible_reward += 0.1
        else:
            possible_reward -= 0.05
    return possible_reward
