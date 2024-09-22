import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_FILE = 'tpl_based_agent.pt'

# FEATURE VECTOR 
#     coin direction 
#     bomb availiable  
#     in danger 
#     escape direction
#     crate nearby


def setup(self):
    """
    Setup your code. This is called once when loading each agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile(MODEL_FILE):
        self.logger.info("Setting up model from scratch.")
        #sizes
        num_coin_directions = 5  # 4 directions + WAIT
        num_bomb_status = 2  # Bomb available or not
        num_danger_status = 2  # In danger or not
        num_escape_routes = 5  # 4 escape directions + WAIT
        num_crate_nearby = 2  # crate in adjacent block or not
        
        # Initialize the Q-table with the shape that matches the features
        self.model = np.random.rand(num_coin_directions, num_bomb_status, num_danger_status, num_escape_routes, num_crate_nearby, len(ACTIONS))
    else:
        self.logger.info("Loading model from saved state.")
        with open(MODEL_FILE, "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # Convert game state to features
    features = state_to_features(game_state)

    # Exploration vs Exploitation
    # Explore
    if self.train and np.random.rand() < 0.1:  # 10% chance to explore
        self.logger.debug("Choosing action purely at random.")
        random_choice = np.random.choice(ACTIONS)
        self.logger.debug(random_choice)
        return random_choice

    # # TODO: explode boxes
    # if features[1] == 1 and features[-1] == 1 and not features[2]:
    #     self.logger.debug("Placing a bomb to destroy crates.")
    #     return 'BOMB'
    
    # Exploit
    self.logger.debug("Querying model for action.")
    # Use the model (Q-table) to choose the best action based on the current state
    q_values = self.model[tuple(features)]
    # Rule-based check for invalid actions: moving into walls or crates, placing bomb when you have none
    valid_q = valid_actions(features, game_state, q_values)
    # # TODO: collect coins
    # if features[0] != 4 and valid_q[features[0]] != float('-inf'):
    #     self.logger.debug("coin nearby.")
    #     valid_q[features[0]] = float('inf')
    best_action = ACTIONS[np.argmax(valid_q)]

    self.logger.debug(best_action)
    return best_action


def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e.
    a feature vector.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    x, y = game_state['self'][-1]
    bomb_available = int(game_state['self'][2])
    coin_direction = find_coin_direction(game_state, x, y)
    in_danger, escape_direction = check_danger(game_state, x, y)
    crate_nearby = check_crates(game_state, x, y)

    return (coin_direction, bomb_available, in_danger, escape_direction, crate_nearby)


def find_coin_direction(game_state, x, y):
    """
    Determine the direction of the nearest coin.
    
    :param game_state: The current game state
    :param x: The x-coordinate of the agent
    :param y: The y-coordinate of the agent
    :return: An integer representing the direction (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT, 4: WAIT)
    """
    coins = game_state['coins']
    if not coins:
        return 4    # no coins availiable, use WAIT as placeholder
    
    # Calculate Manhattan distance to each coin
    min_distance = float('inf')
    direction = 4
    for (coin_x, coin_y) in coins:
        distance = abs(coin_x - x) + abs(coin_y - y)
        if distance < min_distance:
            min_distance = distance
            
            # Determine the direction to the nearest coin
            # UP
            if coin_y < y and coin_x == x:
                if not path_blocked('UP', game_state):
                    direction = 0
                else:
                    direction = random.choice((1, 3))
            # RIGHT
            elif coin_y == y and coin_x > x:
                if not path_blocked('RIGHT', game_state):
                    direction = 1
                else:
                    direction = random.choice((0, 2))
            # DOWN
            elif coin_y > y and coin_x == x:
                if not path_blocked('DOWN', game_state):
                    direction = 2
                else:
                    direction = random.choice((1, 3))
            # LEFT
            elif coin_y == y and coin_x < x:
                if not path_blocked('LEFT', game_state):
                    direction = 3
                else:
                    direction = random.choice((0, 2))
            # UP LEFT
            elif coin_y < y and coin_x < x:
                if not path_blocked('UP', game_state):
                    direction = 0
                else:
                    direction = 3
            # UP RIGHT
            elif coin_y < y  and coin_x > x:
                if not path_blocked('UP', game_state):
                    direction = 0
                else:
                    direction = 1
            # down left
            elif coin_y > y and coin_x < x:
                if not path_blocked('DOWN', game_state):
                    direction = 2
                else:
                    direction = 3
            # down right
            elif coin_y > y and coin_x > x:
                if not path_blocked('DOWN', game_state):
                    direction = 2
                else:
                    direction = 1
    return direction

def check_danger(game_state, x, y):
    """
    Check if the agent is in danger (in the blast radius of a bomb) and identify escape routes.
    
    :param game_state: The current game state
    :param x: The x-coordinate of the agent
    :param y: The y-coordinate of the agent
    :return: a tuple: (1 if in danger 0 if not , an integer representing the escape direction (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT))
    """
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    field = game_state['field']
    in_danger = 0
    escape_direction = 4  # Default to WAIT

    for (bomb_x, bomb_y), timer in bombs:
        if timer <= 2:  # Check bombs that are about to explode
            if x == bomb_x and abs(y - bomb_y) <= 3:
                in_danger = 1
            if y == bomb_y and abs(x - bomb_x) <= 3:
                in_danger = 1

    if in_danger:
        # Determine a safe direction
        if y > 0 and field[x, y-1] == 0 and explosion_map[x, y-1] == 0 and not path_blocked(0, game_state):  # UP
            escape_direction = 0
        elif y < len(field[0]) - 1 and field[x, y+1] == 0 and explosion_map[x, y+1] == 0 and not path_blocked(2, game_state):  # DOWN
            escape_direction = 2
        elif x < len(field) - 1 and field[x+1, y] == 0 and explosion_map[x+1, y] == 0 and not path_blocked(1, game_state):  # RIGHT
            escape_direction = 1
        elif x > 0 and field[x-1, y] == 0 and explosion_map[x-1, y] == 0 and not path_blocked(3, game_state):  # LEFT
            escape_direction = 3

    return in_danger, escape_direction

def check_crates(game_state, x, y):
    """
    Checks if there are any crates in the adjacent tiles.
    
    :param game_state: The current game state
    :param x: The x-coordinate of the agent
    :param y: The y-coordinate of the agent
    :return: 1 if there is at least one crate nearby, 0 otherwise
    """
    field = game_state['field']
    
    if y > 0 and field[x, y-1] == 1:  # UP
        return 1
    if y < field.shape[1] - 1 and field[x, y+1] == 1:  # DOWN
        return 1
    if x > 0 and field[x-1, y] == 1:  # LEFT
        return 1
    if x < field.shape[0] - 1 and field[x+1, y] == 1:  # RIGHT
        return 1
    return 0

def path_blocked(action, game_state):
    """
    Checks if the given action would lead the agent into a wall.
    
    :param action: The action to evaluate
    :param game_state: The current game state
    :return: True if the action would lead into a wall, False otherwise
    """
    x, y = game_state['self'][-1]
    field = game_state['field']
    explosion_map = game_state['explosion_map']

    if action == 'UP':
        return y > 0 and (field[x, y-1] != 0 or explosion_map[x, y-1] != 0)
    elif action == 'DOWN':
        return y < field.shape[1] - 1 and (field[x, y+1] != 0 or explosion_map[x, y+1] != 0)
    elif action == 'LEFT':
        return x > 0 and (field[x-1, y] != 0 or explosion_map[x-1, y] != 0)
    elif action == 'RIGHT':
        return x < field.shape[0] - 1 and (field[x+1, y] != 0 or explosion_map[x+1, y] != 0)
    return False

def valid_actions(features, game_state, q_values):
    valid = []
    for i in range(4):  # check blocked movements
        if path_blocked(ACTIONS[i], game_state):
            valid.append(float('-inf'))
        else:
            valid.append(q_values[i])
    valid.append(q_values[4])  # WAIT is always valid
    # TODO: No bombs for coin collector
    if features[1] == 0:  # cant place bomb if you dont have one
        valid.append(float('-inf'))
    else:
        valid.append(float('-inf'))
    return valid