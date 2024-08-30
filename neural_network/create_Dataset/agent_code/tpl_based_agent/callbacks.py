import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

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
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        #sizes
        num_coin_directions = 5  # 4 directions + WAIT
        num_bomb_status = 2  # Bomb available or not
        num_danger_status = 2  # In danger or not
        num_escape_routes = 5  # 4 escape directions + WAIT
        num_crate_nearby = 2  # crate in adjacent block or not
        
        # Initialize the Q-table with the shape that matches the features
        self.model = np.zeros((num_coin_directions, num_bomb_status, num_danger_status, num_escape_routes, num_crate_nearby, len(ACTIONS)))
    else:
        self.logger.info("Loading model from saved state.")
        with open("tpl_based_agent.pt", "rb") as file:
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
    self.logger.debug("Querying model for action.")
    # Use the model (Q-table) to choose the best action based on the current state
    q_values = self.model[tuple(features)]
    best_action = ACTIONS[np.argmax(q_values)]

    # TODO: avoid walls, explode boxes

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
    # TODO
    return 4 

def check_danger(game_state, x, y):
    # TODO
    return 0, 0

def check_crates(game_state, x, y):
    # TODO
    return 0