import torch
import torch.nn as nn
import torch.optim as optim
from agent_code.TRAIN_3Conv1Hidden1Subsampling.rewards import reward


def setup_training(self):
    self.transitions = []

    return 0


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    self.transitions.append([old_game_state, self_action, new_game_state])
    return 0


def end_of_round(self, last_game_state, last_action, events):
    self.optimizer = optim.Adam(self.model.parameters, lr=0.001)
    self.criterion = nn.MSELoss()

    for old_state, action, new_state in self.transitions[:-1]:
        # Convert states to tensors and reshape for the network
        old_state_field = torch.tensor(self.rewriteGameState(old_state), dtype=torch.float32).unsqueeze(0).unsqueeze(
            0)  # [1, 1, 17, 17]
        new_state_field = torch.tensor(self.rewriteGameState(new_state), dtype=torch.float32).unsqueeze(0).unsqueeze(
            0)  # [1, 1, 17, 17]

        # Get current Q-values for the old state
        q_values = self.model.convolution_model(old_state_field)
        target_q_values = q_values.clone()

        # Get the reward for the transition
        old_state_reward = reward(old_state, action)

        with torch.no_grad():
            # Get max Q-value for the next state
            next_q_values = self.model.convolution_model(new_state_field)
            max_next_q_value = torch.max(next_q_values)

        # Update Q-value for the taken action
        target_q_value = old_state_reward + 0.05 * max_next_q_value
        target_q_values[0, self.actions.index(action)] = target_q_value
        # Perform Q-learning update
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    weights = {}
    weights["w_conv1"] = self.model.parameters[0]
    weights["w_conv2"] = self.model.parameters[1]
    weights["w_conv3"] = self.model.parameters[2]
    weights["w_conv4"] = self.model.parameters[3]
    weights["w_conv5"] = self.model.parameters[4]
    weights["w_conv6"] = self.model.parameters[5]
        # hidden layer with 2048 input and 256 output neurons
    weights["w_h1"] = self.model.parameters[-2]
    weights["w_o"] = self.model.parameters[-1]
    torch.save(weights, "weights.pth")

    # Clear transitions after the round ends
    self.transitions.clear()
    return 0
