import os
import pickle
import random
import heapq
from collections import deque

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DIRECTION_OFFSETS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
MODEL_FILE = 'my_box_agent_6.pt'
GRAPH_ROUNDS = []
GRAPH_STEPS = []
GRAPH_STEPS_MEAN = []
GRAPH_SCORE = []
GRAPH_SCORE_MEAN = []

# FEATURE VECTOR 
#     coin direction 
#     bomb safe  
#     in danger 
#     escape direction

#TODO 
def setup(self):
    """
    Setup your code. This is called once when loading each agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    if self.train or not os.path.isfile(MODEL_FILE):
        self.logger.info("Setting up model from scratch.")
        #sizes
        num_coin_directions = 5  # 4 directions + WAIT
        num_bomb_status = 2  # Safe to place bomb or not
        num_danger_status = 2  # In danger or not
        num_escape_routes = 5  # 4 escape directions + WAIT
        
        # Initialize the Q-table with the shape that matches the features
        self.model = np.random.rand(num_coin_directions, num_bomb_status, num_danger_status, num_escape_routes, len(ACTIONS))
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
    
    # Exploit
    # if self.train:
    # Rule: explode boxes
    if features[1] == 1 and crate_nearby(game_state) and not features[2]: 
        self.logger.debug("Placing a bomb to destroy crates.")
        return 'BOMB'
    # Rule: must escape if in danger
    if features[2]:
        return ACTIONS[features[3]]

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
    Converts the game state to the input of the model, i.e.
    a feature vector.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    if game_state is None:
        return None

    x, y = game_state['self'][-1]
    bomb_available = int((game_state['self'][2] and find_escape(game_state, 5) != None and crate_nearby(game_state)))
    coin_direction = find_coin_direction_dijkstra(game_state, x, y)
    in_danger, bomb_direction = check_danger(game_state, x, y)
    if in_danger:
        escape_direction = find_escape(game_state, bomb_direction)
    else:
        escape_direction = 4

    return (coin_direction, bomb_available, in_danger, escape_direction)


def check_danger(game_state, x, y):
    """
    Check if the agent is in danger (in the blast radius of a bomb) and identify escape routes.
    
    :param game_state: The current game state
    :param x: The x-coordinate of the agent
    :param y: The y-coordinate of the agent
    :return: a tuple: (1 if in danger 0 if not , an integer representing the escape direction (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT))
    """
    bombs = game_state['bombs']
    in_danger = 0
    bomb_direction = 4  # Default to WAIT

    for (bomb_x, bomb_y), t in bombs:
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

    return in_danger, bomb_direction


def find_escape(game_state, bomb_direction):
    """
    find an escape direction
    
    :param game_state: The current game state
    :param bomb_direction: direction of the bomb: 0: above, 1: to right, 2: below, 3: to left, 4: default, 5: under me
    :return: an integer representing the escape direction (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT))
    """
    # Determine a safe direction
    # use mod 4 to find direction preferencing a corner
    x, y = game_state['self'][-1]
    if bomb_direction == 5:
        ideal_directions = no_deadends(game_state)
        if ideal_directions:
        # TODO: can make a better ideal direction based on coin proximity
            return random.choice(ideal_directions)
    else:
        ideal_directions = [(bomb_direction + 1)%4, (bomb_direction + 3)%4, (bomb_direction + 2)%4]
        for direction in ideal_directions:
            if not path_blocked(direction, (x, y), game_state):
                return direction
    return 4
    

def no_deadends(game_state):
    """
    check each direction from current position to see if there is a 
    dead end (no escape if a bomb was placed now)
    
    :param game_state: The current game state
    :return: a list of integers representing directions that are not a dead end (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT))
    """
    x, y = game_state['self'][-1]
    safe_directions = []

    for dir in range(4):
        blocked = 0
        for distance in range(4):
            new_offset = (DIRECTION_OFFSETS[dir][0] * distance, DIRECTION_OFFSETS[dir][1] * distance)
            new_x = x + new_offset[0]
            new_y = y + new_offset[1]
            directions = [dir, (dir + 1) % 4, (dir - 1) % 4] # [straight, corner, other corner]
            blockage = [path_blocked(directions[0], (new_x, new_y), game_state), path_blocked(directions[1], (new_x, new_y), game_state), path_blocked(directions[2], (new_x, new_y), game_state)]
            if distance == 0 and blockage[0]:
                blocked = 1
                break
            elif distance > 0 and (not blockage[1] or not blockage[2]):
                break
            elif blockage[0]:
                blocked = 1
                break
        if not blocked: safe_directions.append(dir)

    return safe_directions


def crate_nearby(game_state):
    """
    Checks if there are any crates in the adjacent tiles.
    
    :param game_state: The current game state
    :return: 1 if there is at least one crate nearby, 0 otherwise
    """
    field = game_state['field']
    x, y = game_state['self'][-1]
    
    if y >= 0 and field[x, y-1] == 1:  # UP
        return 1
    if y <= field.shape[1] - 1 and field[x, y+1] == 1:  # DOWN
        return 1
    if x >= 0 and field[x-1, y] == 1:  # LEFT
        return 1
    if x <= field.shape[0] - 1 and field[x+1, y] == 1:  # RIGHT
        return 1
    return 0


def path_blocked(action, position, game_state) -> int:
    """
    Checks if the given action would lead the agent into a wall.
    
    :param action: The action to evaluate
    :param position: Tuple (x, y) of the position from which to check from
    :param game_state: The current game state
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
        d1, b1 = check_danger(game_state, x, y)
        y -= 1
        d2, b2 = check_danger(game_state, x, y)
        if (y >= 0      # check new coordinate is valid
            and (field[x, y] == -1      # wall in the way
                 or explosion_map[x, y] != 0    # current explosion
                 or (x, y) in bomb_locations    # bomb in the way
                 or (d2 and not d1))):      # am not in danger but would move into danger (danger -> danger is fine)
            return 1
        elif y >= 0 and field[x, y] == 1:   # crate in the way
            return 2
    elif action == 2:   # DOWN
        d1, b1 = check_danger(game_state, x, y)
        y += 1
        d2, b2 = check_danger(game_state, x, y)
        if y <= field.shape[1] - 1 and (field[x, y] == -1 or explosion_map[x, y] != 0 or (x, y) in bomb_locations or (d2 and not d1)):
            return 1
        elif y <= field.shape[1] - 1 and field[x, y] == 1:
            return 2
    elif action == 3:   # LEFT
        d1, b1 = check_danger(game_state, x, y)
        x -= 1
        d2, b2 = check_danger(game_state, x, y)
        if x >= 0 and (field[x, y] == -1 or explosion_map[x, y] != 0 or (x, y) in bomb_locations or (d2 and not d1)):
            return 1
        elif x >= 0 and field[x, y] == 1:
            return 2
    elif action == 1:   # RIGHT
        d1, b1 = check_danger(game_state, x, y)
        x += 1
        d2, b2 = check_danger(game_state, x, y)
        if x <= field.shape[0] - 1 and (field[x, y] == -1 or explosion_map[x, y] != 0 or (x, y) in bomb_locations or (d2 and not d1)):
            return 1
        elif x <= field.shape[0] - 1 and field[x, y] == 1:
            return 2
    
    return 0


def valid_actions(features, game_state, q_values):
    """
    Checks if an action is valid
    
    :param features: feature vector
    :param game_state: The current game state
    :param q_values: the model
    :return: the model but all invalid actions are changed to -infinity
    """
    position = game_state['self'][-1]
    valid = []
    for i in range(4):  # check blocked movements
        if path_blocked(i, position, game_state):
            valid.append(float('-inf'))
        else:
            valid.append(q_values[i])
    valid.append(float(4))  # WAIT is always valid
    if features[1] == 1:  # only place bombs if you have one
        valid.append(q_values[5])
    else:
        valid.append(float('-inf'))
    return valid
    

def find_coin_direction_dijkstra(game_state, x, y):
    """
    Find the direction to the nearest coin using Dijkstra's algorithm.
   
    :param game_state: The current game state.
    :param x: The x-coordinate of the agent.
    :param y: The y-coordinate of the agent.
    :return: The direction (UP, RIGHT, DOWN, LEFT, WAIT) to the nearest coin.
    """
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
            for i in range(4):
                new_x, new_y = x + DIRECTION_OFFSETS[i][0], y + DIRECTION_OFFSETS[i][1]
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

    coins = game_state['coins']
    if not coins:
        return find_crate_direction(game_state, x, y)   # no coins availiable so find closest crate 
        valid_actions = [4]
        for i in range(4):
            if not path_blocked(i, game_state['self'][-1], game_state):
                valid_actions.append(i)
        return np.random.choice(valid_actions)    # no coins availiable, use WAIT as placeholder
    dir = dijkstra(game_state, x, y)
    if dir ==4: return find_crate_direction(game_state, x, y)
    return dir


def find_crate_direction(game_state, start_x, start_y):
    """
    Find the direction to the nearest crate using BFS.
   
    :param game_state: The current game state.
    :param x: The x-coordinate of the agent.
    :param y: The y-coordinate of the agent.
    :return: The direction (UP, RIGHT, DOWN, LEFT, WAIT) to the nearest crate.
    """
    
    q = deque() # (direction, x, y)
    for i in range(4): # add up right down left to queue
        q.append((i, start_x + DIRECTION_OFFSETS[i][0], start_y + DIRECTION_OFFSETS[i][1]))

    visited = set()   # the field we are on cannot be a crate
    visited.add((start_x, start_y))
    field = game_state['field']

    while q:
        first_direction, x, y = q.popleft()
        visited.add((x, y))

        if 0 < x < field.shape[0] and 0 < y < field.shape[1]:
            # If this tile is a crate, return the first direction
            if field[x, y] == 1:
                return first_direction    

            # Explore neighbors (UP, RIGHT, DOWN, LEFT)
            for i in range(4):
                new_x, new_y = x + DIRECTION_OFFSETS[i][0], y + DIRECTION_OFFSETS[i][1]
                # Check if the neighbor has not been visited and is a free tile
                if (new_x, new_y) not in visited:
                    blocked = path_blocked(i, (x, y), game_state)
                    if blocked != 1:
                        q.append((first_direction, new_x, new_y))
                
    return 4  # wait if no way to a crate