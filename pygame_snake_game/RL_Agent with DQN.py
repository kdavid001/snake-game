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
STATE_SIZE = 4  # grid_x, grid_y, food_dir, danger_level
BATCH_SIZE = 64
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

# csv save function
def save_weights_to_csv(state_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for key, weight in state_dict.items():
            writer.writerow([key])
            flat_weights = weight.flatten().tolist()
            writer.writerow(flat_weights)
            writer.writerow([])
    print(f"Weights saved to {path}")

class DQN(nn.Module):
    """Deep Q-Network with state representation"""
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.fc(x)


class DQNAgent:
    def __init__(self):
        self.policy_net = DQN(STATE_SIZE, 4)
        self.target_net = DQN(STATE_SIZE, 4)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.steps_done = 0

        # Initialize target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def get_state(self, game_state):
        """state representation with full danger detection"""
        head = game_state['snake_head']
        food = game_state['food']
        body = game_state['snake_body']

        # Normalized grid position
        grid_x = head[0] // BLOCK_SIZE
        grid_y = head[1] // BLOCK_SIZE

        # Food direction (0-3: up, right, down, left)
        dx, dy = food[0] - head[0], food[1] - head[1]
        food_dir = (1 if dx > 0 else 3) if abs(dx) > abs(dy) else (2 if dy > 0 else 0)

        # Danger detection in all 4 directions
        danger = 0
        # Check left, right, up, down
        for x, y in [
            (head[0] - BLOCK_SIZE, head[1]),  # Left
            (head[0] + BLOCK_SIZE, head[1]),  # Right
            (head[0], head[1] - BLOCK_SIZE),  # Up
            (head[0], head[1] + BLOCK_SIZE)  # Down
        ]:
            if (x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT) or ((x, y) in body):
                danger += 1

        # Normalized danger level (0-1.0)
        danger_level = danger / 4

        return torch.FloatTensor([
            grid_x / GRID_WIDTH,
            grid_y / GRID_HEIGHT,
            food_dir / 3,
            danger_level
        ])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)

        with torch.no_grad():
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * GAMMA * next_q

        # Compute loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

        # Update target network periodically
        if self.steps_done % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.steps_done += 1


# Training Setup
agent = DQNAgent()
scores = []
mean_scores = []
best_mean_score = float('-inf')

# Model Loading
WEIGHT_PATH = 'weight file for DQN/snake_dqn.pth'
if os.path.exists(WEIGHT_PATH):
    agent.policy_net.load_state_dict(torch.load(WEIGHT_PATH), weights_only = True)
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    print("Loaded saved weights")

# Training Loop
for episode in range(5000):
    state = game.reset()
    current_state = agent.get_state(state)
    total_reward = 0
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                torch.save(agent.policy_net.state_dict(), WEIGHT_PATH)
                print("Saved weights in .pth")
                pygame.quit()
                sys.exit()

        action = agent.act(current_state)
        next_state, reward, done = game.step(action)
        next_state_processed = agent.get_state(next_state)

        # Store experience with negative reward for collisions
        agent.remember(current_state, action, reward, next_state_processed, done)
        agent.learn()

        current_state = next_state_processed
        total_reward += reward

        # Rendering
        game.render(screen, clock.get_fps())
        pygame.display.flip()
        clock.tick(120)  # Reduce speed for better observation

    # Episode statistics
    scores.append(total_reward)
    mean_score = np.mean(scores[-100:])
    mean_scores.append(mean_score)

    # Save best model
    if mean_score > best_mean_score:
        best_mean_score = mean_score
        torch.save(agent.policy_net.state_dict(), WEIGHT_PATH)
        print("Saved new weights")

    # Save weights to CSV every 500 episodes
    if episode % 500 == 0 and episode != 0:
        save_weights_to_csv(agent.policy_net.state_dict(), "csv files/DQN-Weights.csv")

    print(f"Ep {episode:04d} | Score: {total_reward:3.0f} | ε: {agent.epsilon:.3f} | Mean: {mean_score:.1f}")


# Final Save at the 5000 episode
torch.save(agent.policy_net.state_dict(), WEIGHT_PATH)
print("Saved dqn.pth")
pygame.quit()

# Note Csv files of the weights are large
# Save to CSV
save_weights_to_csv(agent.policy_net.state_dict(), "csv files/DQN-Weights.csv")
print("Saved weights to scv file")

pygame.quit()