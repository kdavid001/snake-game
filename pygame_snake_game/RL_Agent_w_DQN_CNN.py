"""
This is just a copy of the RL_Agent_with_DQN.py file but
with advanced Observation space using CNNs
"""
import numpy as np
import random
import pygame
import sys
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from snake_game import SnakeGame
import csv

# Game Constants
WIDTH, HEIGHT = 800, 600
BLOCK_SIZE = 20
GRID_WIDTH = WIDTH // BLOCK_SIZE
GRID_HEIGHT = HEIGHT // BLOCK_SIZE

# Neural Network Parameters
# STATE_SIZE = 4  # grid_x, grid_y, food_dir, danger_level
# STATE_SIZE = 39  # 14 + local grid (5 x 5)
STATE_SIZE = 59  # 10 + local grid (7 x 7) removed the danger setup
BATCH_SIZE = 128
MEMORY_SIZE = 10000
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001

# Initialize Game
game = SnakeGame(width=WIDTH, height=HEIGHT)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# csv save function
# def save_weights_to_csv(state_dict, path):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     with open(path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         for key, weight in state_dict.items():
#             writer.writerow([key])
#             flat_weights = weight.flatten().tolist()
#             writer.writerow(flat_weights)
#             writer.writerow([])
#     print(f"Weights saved to {path}")


class DQN(nn.Module):
    """Deep Q-Network with state representation"""

    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # input: 1x7x7 → output: 16x5x5
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # output: 32x7x7
            nn.ReLU(),
            nn.Flatten(),  # 32 * 7 * 7 = 800
        )
        self.fc = nn.Sequential(
            nn.Linear(1568 + (input_size - 49), 128),  # 25 grid + rest = input_size
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    # def forward(self, x):
    #     # Split into grid and flat part
    #     grid = x[:, -25:].view(-1, 1, 5, 5)  # last 25 values as 5x5 grid
    #     other = x[:, :-25]
    #     grid_features = self.cnn(grid)
    #     combined = torch.cat((grid_features, other), dim=1)
    #     return self.fc(combined)

    def forward(self, x):
        # Split into grid and flat part
        grid = x[:, -49:].view(-1, 1, 7, 7)  # last 25 values as 5x5 grid
        other = x[:, :-49]
        grid_features = self.cnn(grid)
        combined = torch.cat((grid_features, other), dim=1)
        return self.fc(combined)


class DQNAgent:
    def __init__(self):
        # Policy Network
        self.policy_net = DQN(STATE_SIZE, 4)  # Policy_Network
        self.target_net = DQN(STATE_SIZE, 4)  # Target_network

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.steps_done = 0

        # Initialize target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def get_state(self, game_state):
        head = game_state['snake_head']
        food = game_state['food']
        body = set(game_state['snake_body'])
        direction = game_state['direction']

        # Normalize head and food position
        grid_x = head[0] / WIDTH
        grid_y = head[1] / HEIGHT
        dx = (food[0] - head[0]) / WIDTH
        dy = (food[1] - head[1]) / HEIGHT

        # Direction one-hot
        dir_up = 1 if direction == (0, -BLOCK_SIZE) else 0
        dir_right = 1 if direction == (BLOCK_SIZE, 0) else 0
        dir_down = 1 if direction == (0, BLOCK_SIZE) else 0
        dir_left = 1 if direction == (-BLOCK_SIZE, 0) else 0

        # Danger checks
        # danger_left = int((head[0] - BLOCK_SIZE, head[1]) in body or head[0] - BLOCK_SIZE < 0)
        # danger_right = int((head[0] + BLOCK_SIZE, head[1]) in body or head[0] + BLOCK_SIZE >= WIDTH)
        # danger_up = int((head[0], head[1] - BLOCK_SIZE) in body or head[1] - BLOCK_SIZE < 0)
        # danger_down = int((head[0], head[1] + BLOCK_SIZE) in body or head[1] + BLOCK_SIZE >= HEIGHT)

        # Snake length (normalized)
        max_len = (GRID_WIDTH * GRID_HEIGHT) - 1
        snake_len_norm = len(body) / max_len

        # 5x5 local grid: 1 if wall or body, 0 if free
        half_window = 3
        local_grid = []
        for dy_offset in range(-half_window, half_window + 1):
            for dx_offset in range(-half_window, half_window + 1):
                check_x = head[0] + dx_offset * BLOCK_SIZE
                check_y = head[1] + dy_offset * BLOCK_SIZE
                if (check_x < 0 or check_x >= WIDTH or check_y < 0 or check_y >= HEIGHT
                        or (check_x, check_y) in body):
                    local_grid.append(1.0)
                else:
                    local_grid.append(0.0)

        # Final state vector
        state = [
                    grid_x, grid_y,
                    dx, dy,
                    # danger_left, danger_right, danger_up, danger_down,
                    dir_up, dir_right, dir_down, dir_left,
                    snake_len_norm,
                    1.0  # constant bias term
                ] + local_grid

        return torch.FloatTensor(state)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)

        with torch.no_grad():
            q_values = self.policy_net(state.unsqueeze(0))  # Make it shape (1, STATE_SIZE)
            return q_values.argmax().item()

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        # unpacks them into separate lists
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors and move to GPU
        states = torch.stack(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.stack(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            # next_q = self.target_net(next_states).max(1)[0]
            # Double DQN
            next_actions = self.policy_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + (1 - dones) * GAMMA * next_q

        # Compute loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        # Back propagation
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

        # Update target network periodically
        if self.steps_done % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.steps_done += 1


# Training Setup moving to "GPU"
agent = DQNAgent()
agent.policy_net.to(device)
agent.target_net.to(device)
scores = []
mean_scores = []
episodes = []
best_mean_score = float('-inf')

# TODO: Change the file name
# Model Loading
NEW_WEIGHT_PATH = "Current DQN WEIGHTS/CNN_weights(7x7).pth"  # higheset score :
RETRAIN = True
if os.path.exists(NEW_WEIGHT_PATH):
    # soon In pytouch, this code below would not be able to run without this {weights_only = True}, check for the
    # updates overtime.
    # agent.policy_net.load_state_dict(torch.load(WEIGHT_PATH), weights_only = True)
    agent.epsilon = 0.2 if RETRAIN else EPSILON_END
    agent.policy_net.load_state_dict(torch.load(NEW_WEIGHT_PATH))
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    print("Loaded saved weights")


# Training Loop
def train_model():
    global best_mean_score

    for episode in range(3000):
        state = game.reset()
        current_state = agent.get_state(state).to(device)
        total_reward = 0
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # torch.save(agent.policy_net.state_dict(), WEIGHT_PATH)
                    # print("Saved weights in .pth")
                    pygame.quit()
                    sys.exit()

            action = agent.act(current_state)
            next_state, reward, done = game.step(action)
            # reward = reward.to(device)
            next_state_processed = agent.get_state(next_state).to(device)  # get the processed state and move to GPU

            # Store experience with negative reward for collisions
            agent.remember(current_state, action, reward, next_state_processed, done)
            agent.learn()

            current_state = next_state_processed
            total_reward += reward

            # Rendering
            game.render(screen, clock.get_fps())
            pygame.display.flip()
            clock.tick(240)  # Reduce speed for better observation

        # Episode statistics
        scores.append(total_reward)
        mean_score = np.mean(scores[-100:])
        mean_scores.append(mean_score)
        episodes.append(episode)
        # best_mean_score = 1000 #for 5x5 model with danger direction.
        # Save best model
        if mean_score > best_mean_score:
            best_mean_score = mean_score
            torch.save(agent.policy_net.state_dict(), NEW_WEIGHT_PATH)
            print("Saved new weights")

        # Save weights to CSV every 500 episodes
        # if episode % 500 == 0 and episode != 0:
        #     from matplotlib import pyplot as plt
        #     plt.figure(figsize=(12, 6))
        #     plt.plot(episodes, scores, label="Raw Score per Episode", alpha=0.4)
        #     plt.plot(episodes, mean_scores, label="Mean Score (last 100)", linewidth=2)
        #     plt.title("DQN Training Progress")
        #     plt.xlabel("Episode")
        #     plt.ylabel("Score")
        #     plt.legend()
        #     plt.grid(True)
        #     plt.show()
        current_score = game.get_state()["score"]
        high_score = game.get_state()["highscore"]
        print(f"Ep {episode:04d} | Score: {total_reward:3.0f} | ε: {agent.epsilon:.3f} | Mean: {mean_score:.1f} | "
              f"highscore : {high_score} | current_score: {current_score}")

    # Final Save at the 5000 episode
    # torch.save(agent.policy_net.state_dict(), WEIGHT_PATH)

    # Note Csv files of the weights are large
    # Save to CSV
    # save_weights_to_csv(agent.policy_net.state_dict(), "csv files/DQN-Weights.csv")
    # print("Saved weights to scv file")
    pygame.quit()


if __name__ == '__main__':
    train_model()
    pygame.quit()
