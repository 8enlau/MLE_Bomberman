import os
import pickle
import random
import heapq
from typing import List
import csv

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_FILE = 'my_coin_agent_6-6.pt'

# FEATURE VECTOR 
#     coin direction 
#     bomb availiable  
#     in danger 
#     escape direction
#     crate nearby
GRAPH_ROUNDS = []
GRAPH_STEPS = []
GRAPH_STEPS_MEAN = []
GRAPH_SCORE = []
GRAPH_SCORE_MEAN = []


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
    coin_direction = find_coin_direction_dijkstra(game_state, x, y)
    in_danger, escape_direction = check_danger(game_state, x, y)
    crate_nearby = check_crates(game_state, x, y)

    return (coin_direction, bomb_available, in_danger, escape_direction, crate_nearby)


def find_coin_direction(game_state, x, y):
    """
    Determine the direction of the nearest coin.
    
    :param game_state: The current game state
    :param x: The x-coordinate of the agent
    :param y: The y-coordinate of the agent
    :return: An integer representing the direction (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT, 4: WAIT, 5: UP LEFT, 6: UP RIGHT, 7: DOWN LEFT 8: DOWN RIGHT)
    """
    coins = game_state['coins']
    if not coins:
        return 4    # no coins availiable, use WAIT as placeholder
    
    # Calculate Manhattan distance to each coin
    min_distance = float('inf')
    closest = [0, 0, 0, 0]  # UP, RIGHT, DOWN, LEFT
    direction = 4
    for (coin_x, coin_y) in coins:
        distance_y = abs(coin_y - y)
        distance_x = abs(coin_x - x)
        distance = distance_y + distance_x
        if distance < min_distance:
            min_distance = distance
            # Determine the direction to the nearest coin
            # UP
            if coin_y <= y:
                closest[0] = distance_y
            # RIGHT
            if coin_x >= x:
                closest[1] = distance_x
            # DOWN
            elif coin_y >= y:
                closest[2] = distance_y
            # LEFT
            elif coin_x <= x:
                closest[3] = distance_x
    while closest[np.argmax(closest)] != 0:
        direction = np.argmax(closest)
        if not path_blocked(ACTIONS[direction], (x, y), game_state, True):
            return direction
        else:
            closest[direction] = 0
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
        if y > 0 and field[x, y-1] == 0 and explosion_map[x, y-1] == 0 and not path_blocked(0, (x, y), game_state, True):  # UP
            escape_direction = 0
        elif y < len(field[0]) - 1 and field[x, y+1] == 0 and explosion_map[x, y+1] == 0 and not path_blocked(2, (x, y), game_state, True):  # DOWN
            escape_direction = 2
        elif x < len(field) - 1 and field[x+1, y] == 0 and explosion_map[x+1, y] == 0 and not path_blocked(1, (x, y), game_state, True):  # RIGHT
            escape_direction = 1
        elif x > 0 and field[x-1, y] == 0 and explosion_map[x-1, y] == 0 and not path_blocked(3, (x, y), game_state, True):  # LEFT
            escape_direction = 3

    return in_danger, escape_direction

def check_crates(game_state, x, y):
    """
    Checks if there are anew_y crates in the adjacent tiles.
    
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

def path_blocked(action, position, game_state, boxes_block):
    """
    Checks if the given action would lead the agent into a wall.
    
    :param action: The action to evaluate
    :param position: Tuple (x, y) of the position from which to check from
    :param game_state: The current game state
    :param boxes_block: Whether of not boxes are counted as a block
    :return: True if the action would lead into a wall, False otherwise
    """
    x, y = position
    field = game_state['field']
    explosion_map = game_state['explosion_map']
    
    if boxes_block:
        if action == 'UP':
            return y > 0 and (field[x, y-1] != 0 or explosion_map[x, y-1] != 0)
        elif action == 'DOWN':
            return y < field.shape[1] - 1 and (field[x, y+1] != 0 or explosion_map[x, y+1] != 0)
        elif action == 'LEFT':
            return x > 0 and (field[x-1, y] != 0 or explosion_map[x-1, y] != 0)
        elif action == 'RIGHT':
            return x < field.shape[0] - 1 and (field[x+1, y] != 0 or explosion_map[x+1, y] != 0)
        return False
    else:
        if action == 'UP':
            return y > 0 and (field[x, y-1] == -1 or explosion_map[x, y-1] != 0)
        elif action == 'DOWN':
            return y < field.shape[1] - 1 and (field[x, y+1] == -1 or explosion_map[x, y+1] != 0)
        elif action == 'LEFT':
            return x > 0 and (field[x-1, y] == -1 or explosion_map[x-1, y] != 0)
        elif action == 'RIGHT':
            return x < field.shape[0] - 1 and (field[x+1, y] == -1 or explosion_map[x+1, y] != 0)
        return False


def valid_actions(features, game_state, q_values):
    valid = []
    for i in range(4):  # check blocked movements
        if path_blocked(ACTIONS[i], game_state['self'][-1], game_state, True):
            valid.append(float('-inf'))
        else:
            valid.append(q_values[i])
    valid.append(float(4))  # WAIT is always valid
    if features[1] == 0:  # cant place bomb if you dont have one
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
    field = game_state['field']
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
        for i, dir in enumerate(ACTIONS[:4]):
            new_x, new_y = x + direction_offsets[i][0], y + direction_offsets[i][1]
            # Check if the neighbor has not been visited and is a free tile
            if (new_x, new_y) not in visited and not path_blocked(dir, (x, y), game_state, False): # see closest path through boxes
                new_distance = current_distance + 1  # Uniform cost for each move
                heapq.heappush(pq, (new_distance, new_x, new_y))

                # If we've found a shorter path
                if ((new_x, new_y) not in distance_dict or new_distance < distance_dict[(new_x, new_y)]):
                    distance_dict[(new_x, new_y)] = new_distance
                    parent_dict[(new_x, new_y)] = (x, y, i)  # Store the predecessor and direction
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
        return 4    # no coins availiable, use WAIT as placeholder
    dir = dijkstra(game_state, x, y)
    return dir


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    used for graphing
    """
    steps_survived = last_game_state['step']
    round = last_game_state['round']
    score = last_game_state['self'][1]
    GRAPH_ROUNDS.append(round)
    GRAPH_SCORE.append(score)
    GRAPH_STEPS.append(steps_survived)
    GRAPH_SCORE_MEAN.append(np.mean(GRAPH_SCORE))
    GRAPH_STEPS_MEAN.append(np.mean(GRAPH_STEPS))
    with open('game_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(GRAPH_ROUNDS)
        writer.writerow(GRAPH_SCORE)
        writer.writerow(GRAPH_SCORE_MEAN)
        writer.writerow(GRAPH_STEPS)
        writer.writerow(GRAPH_STEPS_MEAN)
    f.close()    # plot(PLOT_coins, PLOT_mean_coins, "Coins Collected")