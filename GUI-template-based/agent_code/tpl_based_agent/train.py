from collections import namedtuple, deque

import pickle
from typing import List

import events as e
import numpy as np
from .callbacks import state_to_features
from .callbacks import ACTIONS, MODEL_FILE

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
GAMMA = 0.9
ALPHA = 0.1
TRANSITION_HISTORY_SIZE = 1000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
REPETITIVE_ACTION = "REPETITIVE_ACTION"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.last_actions = deque(maxlen=3)  # Track the last 3 actions


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Update the model based on the transition.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)

    action_index = ACTIONS.index(self_action)
    
    # Add the current action to the last_actions deque
    self.last_actions.append(self_action)
    
    # Check for repetitive behavior (e.g., LEFT, RIGHT, LEFT)
    if repetitive_action(self):
        events.append(REPETITIVE_ACTION)
        
    reward = reward_from_events(self, events)
    self.transitions.append(Transition(old_features, self_action, new_features, reward))

    # Q-learning update
    old_q_value = self.model[old_features][action_index]
    future_q_value = np.max(self.model[new_features]) if new_features else 0
    updated_q_value = old_q_value + ALPHA * (reward + GAMMA * future_q_value - old_q_value)
    
    self.model[old_features][action_index] = updated_q_value

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Final update at the end of the round.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step: {last_game_state["step"]}')
    
    last_features = state_to_features(last_game_state)
    action_index = ACTIONS.index(last_action)
    
    reward = reward_from_events(self, events)
    self.transitions.append(Transition(last_features, last_action, None, reward))

    # Final Q-learning update
    old_q_value = self.model[last_features][action_index]
    self.model[last_features][action_index] = old_q_value + ALPHA * (reward - old_q_value)
    
    # Store the model
    with open(MODEL_FILE, "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 20,
        e.WAITED: -5,
        REPETITIVE_ACTION: -10
    }
    
    reward_sum = sum(game_rewards.get(event, 0) for event in events)
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def repetitive_action(self):
     # Check for repetitive behavior (e.g., LEFT, RIGHT, LEFT)
    if len(self.last_actions) == 3:
        if (self.last_actions[0] == 'LEFT' and self.last_actions[1] == 'RIGHT' and self.last_actions[2] == 'LEFT') or \
           (self.last_actions[0] == 'RIGHT' and self.last_actions[1] == 'LEFT' and self.last_actions[2] == 'RIGHT') or \
           (self.last_actions[0] == 'UP' and self.last_actions[1] == 'DOWN' and self.last_actions[2] == 'UP') or \
           (self.last_actions[0] == 'DOWN' and self.last_actions[1] == 'UP' and self.last_actions[2] == 'DOWN'):
           return True