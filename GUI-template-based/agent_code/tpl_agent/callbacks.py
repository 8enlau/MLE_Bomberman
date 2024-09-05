import os
import pickle
import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    """
    if self.train or not os.path.isfile("feature_engineered_model.pt"):
        self.logger.info("Setting up model from scratch.")
        # Define feature space dimensions based on possible feature values
        num_directions = 5  # 4 directions + WAIT
        num_bomb_status = 2  # Bomb available or not
        num_danger_status = 2  # In danger or not
        num_escape_routes = 5  # 4 escape directions + WAIT

        # Initialize the Q-table with the shape that matches the features
        self.model = np.zeros((num_directions, num_bomb_status, num_danger_status, num_escape_routes, len(ACTIONS)))
    else:
        self.logger.info("Loading model from saved state.")
        with open("feature_engineered_model.pt", "rb") as file:
            self.model = pickle.load(file)

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    """
    # Convert game state to features
    features = state_to_features(game_state)

    # Validate feature values to ensure they are within expected ranges
    features = (
        min(max(features[0], 0), 4),  # Ensure nearest_coin_dir is within [0, 4]
        min(max(features[1], 0), 1),  # bomb_available should be within [0, 1]
        min(max(features[2], 0), 1),  # in_danger should be within [0, 1]
        min(max(features[3], 0), 4)   # escape_direction should be within [0, 4]
    )

    # Exploration vs Exploitation
    if self.train and np.random.rand() < 0.1:  # 10% chance to explore
        self.logger.debug("Choosing action purely at random.")
        temp = np.random.choice(ACTIONS)
        self.logger.debug(temp)
        return temp

    print(features[1], features[2])
    # Place a bomb if there's a crate nearby and the agent is not in immediate danger
    # if features[1] == 1 and crates_nearby(game_state) and not features[2]:
    #     self.logger.debug("Placing a bomb to destroy crates.")
    #     return 'BOMB'

    self.logger.debug("Querying model for action.")
    # Use the model (Q-table) to choose the best action based on the current state
    q_values = self.model[tuple(features)]
    print(self.model[tuple(features)])
    best_action = ACTIONS[np.argmax(q_values)]

    # Rule-based check: avoid actions that lead to walls
    while action_leads_to_wall(best_action, game_state):
        self.logger.debug("Best action leads to a wall, choosing alternative.")
        valid_actions = [
            action for action in ACTIONS[:4]  # Only consider movement actions
            if not action_leads_to_wall(action, game_state)
        ]
        q_values[np.argmax(q_values)] = -99
        if not valid_actions:
            best_action = 'WAIT' # No valid actions, WAIT to avoid moving into a wall
        else:
            best_action = ACTIONS[np.argmax(q_values)]

    self.logger.debug(best_action)
    return best_action

def state_to_features(game_state: dict) -> tuple:
    """
    Converts the game state to a tuple of features for use as the Q-table index.
    """
    if game_state is None:
        return None

    # Example feature: (agent position x, agent position y, bomb status, nearest coin direction)
    x, y = game_state['self'][-1]
    bomb_available = int(game_state['self'][2])
    coin_direction = find_coin_direction(game_state, x, y)
    in_danger, escape_direction = check_danger(game_state, x, y)

    return (coin_direction, bomb_available, in_danger, escape_direction)

def find_coin_direction(game_state, x, y):
    """
    Determine the direction of the nearest coin.
    
    :param game_state: The current game state
    :param x: The x-coordinate of the agent
    :param y: The y-coordinate of the agent
    :return: An integer representing the direction (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT)
    """
    coins = game_state['coins']
    if not coins:
        return 4  # No coins available, return a placeholder for WAIT or another strategy
    
    # Calculate Manhattan distance to each coin
    min_distance = float('inf')
    best_direction = 4  # Placeholder for WAIT

    for (coin_x, coin_y) in coins:
        distance = abs(coin_x - x) + abs(coin_y - y)
        
        if distance < min_distance:
            min_distance = distance
            # Determine the direction to the nearest coin
            if coin_y < y:
                best_direction = 0  # UP
            elif coin_y > y:
                best_direction = 2  # DOWN
            elif coin_x > x:
                best_direction = 1  # RIGHT
            elif coin_x < x:
                best_direction = 3  # LEFT

    return best_direction

def action_leads_to_wall(action, game_state):
    """
    Checks if the given action would lead the agent into a wall.
    
    :param action: The action to evaluate
    :param game_state: The current game state
    :return: True if the action would lead into a wall, False otherwise
    """
    x, y = game_state['self'][-1]
    field = game_state['field']

    if action == 'UP':
        return y > 0 and field[x, y-1] == -1  # Walls only, not crates
    elif action == 'DOWN':
        return y < field.shape[1] - 1 and field[x, y+1] == -1  # Walls only, not crates
    elif action == 'LEFT':
        return x > 0 and field[x-1, y] == -1  # Walls only, not crates
    elif action == 'RIGHT':
        return x < field.shape[0] - 1 and field[x+1, y] == -1  # Walls only, not crates
    return False

def crates_nearby(game_state):
    """
    Checks if there are any crates in the adjacent tiles.
    
    :param game_state: The current game state
    :return: True if there is at least one crate nearby, False otherwise
    """
    x, y = game_state['self'][-1]
    field = game_state['field']
    
    if y > 0 and field[x, y-1] == 1:  # UP
        return True
    if y < field.shape[1] - 1 and field[x, y+1] == 1:  # DOWN
        return True
    if x > 0 and field[x-1, y] == 1:  # LEFT
        return True
    if x < field.shape[0] - 1 and field[x+1, y] == 1:  # RIGHT
        return True
    return False

def check_danger(game_state, x, y):
    """
    Check if the agent is in danger (in the blast radius of a bomb) and identify escape routes.
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
        if y > 0 and field[x, y-1] == 0 and explosion_map[x, y-1] == 0 and not action_leads_to_wall(0, game_state):  # UP
            escape_direction = 0
        elif y < len(field[0]) - 1 and field[x, y+1] == 0 and explosion_map[x, y+1] == 0 and not action_leads_to_wall(2, game_state):  # DOWN
            escape_direction = 2
        elif x < len(field) - 1 and field[x+1, y] == 0 and explosion_map[x+1, y] == 0 and not action_leads_to_wall(1, game_state):  # RIGHT
            escape_direction = 1
        elif x > 0 and field[x-1, y] == 0 and explosion_map[x-1, y] == 0 and not action_leads_to_wall(3, game_state):  # LEFT
            escape_direction = 3

    return in_danger, escape_direction