import os
import pickle
import random
import heapq
import sys

import numpy as np
from typing import List



ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_FILE = 'my_box_agent_3.pt'
GRAPH_ROUNDS = []
GRAPH_STEPS = []
GRAPH_SCORE = []

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

    if self.train:
        self.logger.info("Loading coin_basis.pt from saved state.")
        with open("coin_basis.pt", "rb") as file:
            self.model = pickle.load(file)
        sys.stdout = open(os.devnull, 'w')
    elif not os.path.isfile(MODEL_FILE):
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
    
    if self.train:
        # Rule: explode boxes
        if features[1] == 1 and features[-1] == 1 and not features[2] and np.random.rand() < 0.7:  # 70% chance to bomb
            self.logger.debug("Placing a bomb to destroy crates.")
            return 'BOMB'
    # Rule: must escape if in danger
    if features[2]:
        return ACTIONS[features[3]]
    #     # Rule: must move towards coins
    #     if features[0] != 4:
    #         return ACTIONS[features[0]]
    self.logger.debug("Querying model for action.")
    # Use the model (Q-table) to choose the best action based on the current state
    q_values = self.model[tuple(features)]
    # Rule-based check for invalid actions: moving into walls or crates, placing bomb when you have none
    valid_q = valid_actions(features, game_state, q_values)
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
    coin_direction = find_coin_direction_dijkstra(game_state, x, y)
    in_danger, escape_direction = check_danger(game_state, x, y)
    crate_nearby = check_crates(game_state, x, y)

    return (coin_direction, bomb_available, in_danger, escape_direction, crate_nearby)

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
    bomb_direction = 4  # Default to WAIT

    for (bomb_x, bomb_y), timer in bombs:
        if timer:  # Check bombs that are about to explode
            if x == bomb_x and y == bomb_y:
                in_danger = 1
                bomb_direction = 5  # on top of bomb
            elif x == bomb_x and y > bomb_y >= y - 3:
                in_danger = 1
                bomb_direction = 0  # bomb above
            elif x == bomb_x and y < bomb_y <= y + 3:
                in_danger = 1
                bomb_direction = 2  # bomb below
            elif y == bomb_y and x > bomb_x >= x - 3:
                in_danger = 1
                bomb_direction = 3  # bomb to my left
            elif y == bomb_y and x < bomb_x <= x + 3:
                in_danger = 1
                bomb_direction = 1  # bomb to my right

    if in_danger:
        # Determine a safe direction
        # use mod 4 to find direction preferencing a corner
        if bomb_direction == 5:
            ideal_directions = no_deadends(game_state)
            if ideal_directions:
            # TODO: can make a better ideal direction based on coin proximity
                escape_direction = random.choice(ideal_directions)
        else:
            ideal_directions = [(bomb_direction + 1)%4, (bomb_direction + 3)%4, (bomb_direction + 2)%4]
            for direction in ideal_directions:
                if not path_blocked(direction, (x, y), game_state):
                    escape_direction = direction
    return in_danger, escape_direction


def no_deadends(game_state):
    x, y = game_state['self'][-1]
    field = game_state['field']
    direction_offsets = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    safe_directions = []

    for dir in range(4):
        blocked = 0
        for distance in range(4):
            new_offset = (direction_offsets[dir][0] * distance, direction_offsets[dir][1] * distance)
            new_x = x + new_offset[0]
            new_y = y + new_offset[1]
            directions = [dir, (dir + 1) % 4, (dir - 1) % 4]
            if distance > 0 and (not path_blocked(directions[1], (new_x, new_y), game_state)\
                or not path_blocked(directions[2], (new_x, new_y), game_state)):
                continue
            elif not path_blocked(directions[0], (new_x, new_y), game_state):
                pass
            else:
                blocked = 1
                break
        if not blocked: safe_directions.append(dir)
    return safe_directions


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

def path_blocked(action, position, game_state):
    """
    Checks if the given action would lead the agent into a wall.
    
    :param action: The action to evaluate
    :param position: Tuple (x, y) of the position from which to check from
    :param game_state: The current game state
    :param boxes_block: Whether of not boxes are counted as a block
    :return: 0 if not blocked, 1 if wall or explosion, 2 if crate
    """
    x, y = position
    field = game_state['field']
    explosion_map = game_state['explosion_map']
    bombs = game_state['bombs']
    bomb_locations = [bomb[0] for bomb in bombs]
    if x < 0 or x > field.shape[0]-1 or y < 0 or y > field.shape[1]-1:
        return 1
    
    if action == 0: # UP
        y -= 1
        if y > 0 and (field[x, y] == -1 or explosion_map[x, y] != 0 or (x, y) in bomb_locations):
            return 1
        elif y > 0 and field[x, y] == 1:
            return 2
        return (field[x, y-1] == -1 or explosion_map[x, y-1] != 0)
    elif action == 2:   # DOWN
        y += 1
        if y < field.shape[1] - 1 and (field[x, y] == -1 or explosion_map[x, y] != 0 or (x, y) in bomb_locations):
            return 1
        elif y < field.shape[1] - 1 and field[x, y] == 1:
            return 2
    elif action == 3:   # LEFT
        x -= 1
        if x > 0 and (field[x, y] == -1 or explosion_map[x, y] != 0 or (x, y) in bomb_locations):
            return 1
        elif x > 0 and field[x, y] == 1:
            return 2
    elif action == 1:   # RIGHT
        x += 1
        if x < field.shape[0] - 1 and (field[x, y] == -1 or explosion_map[x, y] != 0 or (x, y) in bomb_locations):
            return 1
        elif x < field.shape[0] - 1 and field[x, y] == 1:
            return 2
    
    return 0


def valid_actions(features, game_state, q_values):
    field = game_state['field']
    position = game_state['self'][-1]
    corners = [(1, field.shape[1] - 2), (field.shape[0] - 2, 1), (1, 1), (field.shape[0] - 2, field.shape[1] - 2)]
    valid = []
    for i in range(4):  # check blocked movements
        if path_blocked(i, position, game_state):
            valid.append(float('-inf'))
        else:
            valid.append(q_values[i])
    valid.append(float(4))  # WAIT is always valid
    if features[1] == 0 or position in corners:  # cant place bomb if you dont have one, shouldnt place bomb in start corners
        valid.append(float('-inf'))
    else:
        valid.append(q_values[5])
    return valid


def dijkstra(game_state, start_x, start_y):
    """
    Dijkstra's algorithm to find the shortest path to the nearest coin.
   
    :param game_state: The current game state containing the board and coin positions.
    :param start: The starting position of the agent (x, y).
    :return: The first direction (UP, RIGHT, DOWN, LEFT) to take toward the nearest coin.
    """
    def update_path():
        """helper: If we've found a shorter path update dictionaries"""
        if ((new_x, new_y) not in distance_dict or new_distance < distance_dict[(new_x, new_y)]):
            distance_dict[(new_x, new_y)] = new_distance
            parent_dict[(new_x, new_y)] = (x, y, i)  # Store the predecessor and direction

    coins = game_state['coins']   
    pq = [] # Priority queue (empty heap)
    heapq.heappush(pq, (0, start_x, start_y))  # (distance, x, y) -> distance from agent to itself is 0
    distance_dict = {(start_x, start_y): 0} # shortest known distance to each tile -> distance from agent to itself is 0
    parent_dict = {} # parent of each node for path reconstruction (child x, child y): (parent x, parent y, direction)
    visited = set()

    while pq:
        # Get the tile with the smallest distance
        current_distance, x, y = heapq.heappop(pq)

        # If this tile is a coin, reconstruct the path and return the direction
        if (x, y) in coins:
            return reconstruct_path(parent_dict, (start_x, start_y), (x, y))

        # Mark the current node as visited
        visited.add((x, y))

        # Explore neighbors (UP, RIGHT, DOWN, LEFT)
        direction_offsets = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        for i in range(4):
            new_x, new_y = x + direction_offsets[i][0], y + direction_offsets[i][1]
            # Check if the neighbor has not been visited and is a free tile
            if (new_x, new_y) not in visited:
                blocked = path_blocked(i, (x, y), game_state)
                if not blocked:
                    new_distance = current_distance + 1  # 1 distance for free block
                    heapq.heappush(pq, (new_distance, new_x, new_y))
                    update_path()
                elif blocked == 2: # blocked by crate
                    new_distance = current_distance + 5  # 5 distance for crate (distance penalty to encourage free space movement)
                    heapq.heappush(pq, (new_distance, new_x, new_y))
                    update_path()
    return 4  # wait if no way to a coin


def reconstruct_path(parent_dict, start, end):
    """
    Reconstruct the first move from the start to the end using the parent_dict dictionary.
   
    :param parent_dict: A dictionary mapping each node to its predecessor and direction.
    :param start: The starting position of the agent (x, y).
    :param end: The position of the nearest coin (x, y).
    :return: The first direction (UP, RIGHT, DOWN, LEFT) to take from the start.
    """
    current = end
    # Backtrack from the coin to the start
    while current in parent_dict:
        prev_x, prev_y, direction = parent_dict[current]

        # If the predecessor is the starting point, return the direction to move
        if (prev_x, prev_y) == start:
            return direction

        current = (prev_x, prev_y)
    return 4  # wait if no path is found

def find_coin_direction_dijkstra(game_state, x, y):
    """
    Find the direction to the nearest coin using Dijkstra's algorithm.
   
    :param game_state: The current game state.
    :param x: The x-coordinate of the agent.
    :param y: The y-coordinate of the agent.
    :return: The direction (UP, RIGHT, DOWN, LEFT, WAIT) to the nearest coin.
    """
    coins = game_state['coins']
    if not coins:
        valid_actions = [4]
        for i in range(4):
            if not path_blocked(i, game_state['self'][-1], game_state):
                valid_actions.append(i)
        return np.random.choice(valid_actions)    # no coins availiable, use WAIT as placeholder
    dir = dijkstra(game_state, x, y)
    return dir

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Final update at the end of the round.

    :param self: The same object that is passed to all of your callbacks.
    """
    steps_survived = last_game_state['step']
    print(last_game_state)