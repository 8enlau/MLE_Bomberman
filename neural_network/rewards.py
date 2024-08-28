import random


def reward(situation,action):
    ### Impossible action
    if action_not_possible(situation,action):
        return -1000

    ### Action leads to Dying
    if action_leads_to_suicide(situation,action): # For example blocking self in bomb
        return -1000
    if action_leads_to_dying(situation,action):
        return -800

    ### Being close to a bomb
    # Walking where a bomb will explode soon:
    if in_scope_of_bomb_after_action(situation,action):
        if cannot_escape_after_action(situation,action):
            return -1000


    ### Player can survive, try to maximise gain:
    possible_reward = 0
    if action=="BOMB":
        if bomb_will_kill_opponent(situation):
            possible_reward +=800
        if bomb_might_kill_opponent(situation):
            possible_reward += random.uniform(0,200)
        if bomb_shortens_path_to_coin(situation,action):
            possible_reward += 150
        if bomb_will_destroy_crates(situation):
            possible_reward += 30
        return possible_reward

    if action_leads_to_dying_opponent(situation,action): #For Example standing in the way
                                        # and therefore blocking opponent to stand in bomb.
        possible_reward +=250
    if collecting_coin(situation,action):
        possible_reward +=250
    if walking_closer_to_reachable_coin(situation,action):
        possible_reward +=100