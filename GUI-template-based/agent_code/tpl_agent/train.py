from collections import namedtuple, deque
import numpy as np
import pickle
import events as e

from .callbacks import state_to_features, ACTIONS
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyperparameters
GAMMA = 0.9
ALPHA = 0.1
TRANSITION_HISTORY_SIZE = 1000

def setup_training(self):
    """
    Initialize variables for training.
    """
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list):
    """
    Update the model based on the transition.
    """
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)
    
    # Validate feature values to ensure they are within expected ranges
    old_features = (
        min(max(old_features[0], 0), 4),
        min(max(old_features[1], 0), 1),
        min(max(old_features[2], 0), 1),
        min(max(old_features[3], 0), 4)
    )
    
    new_features = (
        min(max(new_features[0], 0), 4),
        min(max(new_features[1], 0), 1),
        min(max(new_features[2], 0), 1),
        min(max(new_features[3], 0), 4)
    )

    action_index = ACTIONS.index(self_action)
    
    reward = reward_from_events(self, events)
    self.transitions.append(Transition(old_features, self_action, new_features, reward))

    # Q-learning update
    old_q_value = self.model[old_features][action_index]
    future_q_value = np.max(self.model[new_features]) if new_features else 0
    updated_q_value = old_q_value + ALPHA * (reward + GAMMA * future_q_value - old_q_value)
    
    self.model[old_features][action_index] = updated_q_value


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    """
    Final update at the end of the round.
    """
    last_features = state_to_features(last_game_state)

    # Validate feature values to ensure they are within expected ranges
    last_features = (
        min(max(last_features[0], 0), 4),  # nearest_coin_dir should be within [0, 4]
        min(max(last_features[1], 0), 1),  # bomb_available should be within [0, 1]
        min(max(last_features[2], 0), 1),  # in_danger should be within [0, 1]
        min(max(last_features[3], 0), 4)   # escape_direction should be within [0, 4]
    )

    action_index = ACTIONS.index(last_action)
    
    reward = reward_from_events(self, events)
    self.transitions.append(Transition(last_features, last_action, None, reward))

    # Final Q-learning update
    old_q_value = self.model[last_features][action_index]
    self.model[last_features][action_index] = old_q_value + ALPHA * (reward - old_q_value)

    # Save the model after every round
    with open("feature_engineered_model.pt", "wb") as file:
        pickle.dump(self.model, file)

def reward_from_events(self, events: list) -> int:
    """
    Reward function to guide the agent's behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.SURVIVED_ROUND: 2,
        e.KILLED_SELF: -5,
        e.WAITED: -0.1
    }
    
    reward_sum = sum(game_rewards.get(event, 0) for event in events)
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
