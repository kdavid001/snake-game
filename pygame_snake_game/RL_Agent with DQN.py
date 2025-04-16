import numpy as np
import random
import pygame
import sys
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


class DQN(nn.Module):
    """Deep Q-Network with similar state input to  Q-table"""

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
        """ state representation """
        head = game_state['snake_head']
        food = game_state['food']
        body = game_state['snake_body']

        grid_x = max(0, min(head[0] // BLOCK_SIZE, GRID_WIDTH - 1))
        grid_y = max(0, min(head[1] // BLOCK_SIZE, GRID_HEIGHT - 1))

        dx, dy = food[0] - head[0], food[1] - head[1]
        food_dir = (1 if dx > 0 else 3) if abs(dx) > abs(dy) else (2 if dy > 0 else 0)

        danger = 0
        for x, y in [(head[0] - BLOCK_SIZE, head[1]),
                     (head[0], head[1] - BLOCK_SIZE),
                     (head[0] + BLOCK_SIZE, head[1])]:
            if (x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT) or ((x, y) in body):
                danger += 1
        danger_level = min(danger, 2)

        # Normalize state values
        return torch.FloatTensor([
            grid_x / GRID_WIDTH,
            grid_y / GRID_HEIGHT,
            food_dir / 3,
            danger_level / 2
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

        # Update target network
        if self.steps_done % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.steps_done += 1


# Training Loop
agent = DQNAgent()
scores = []
mean_scores = []

for episode in range(5000):
    state = game.reset()
    current_state = agent.get_state(state)
    total_reward = 0
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                torch.save(agent.policy_net.state_dict(), 'weight file for DQN/snake_dqn.pth')
                params = agent.policy_net.state_dict()

                with open("csv files/DQN-Weights.csv", mode='w', newline='') as file:
                    writer = csv.writer(file)
                    for key, weight in params.items():
                        writer.writerow([key])  # Layer name
                        flat_weights = weight.flatten().tolist()
                        writer.writerow(flat_weights)  # Weights in one row
                        writer.writerow([])  # Empty line for readability
                    print("DQN-Weight saved as csv!")

                print("Saved dqn.pth")
                pygame.quit()
                sys.exit()

        action = agent.act(current_state)
        next_state, reward, done = game.step(action)

        # reward structure
        prev_dist = math.dist(state['snake_head'], state['food'])
        new_dist = math.dist(next_state['snake_head'], next_state['food'])
        reward += 0.1 * (prev_dist - new_dist) / BLOCK_SIZE

        next_state_processed = agent.get_state(next_state)

        agent.remember(current_state, action, reward, next_state_processed, done)
        agent.learn()

        current_state = next_state_processed
        total_reward += reward

        # Rendering
        game.render(screen, clock.get_fps())
        pygame.display.flip()
        clock.tick(120)

    # Episode statistics
    scores.append(total_reward)
    mean_scores.append(np.mean(scores[-100:]))

    print(f"Ep {episode:04d} | Total Rewards: {total_reward:3.0f} | ε: {agent.epsilon:.3f} | Mean: {mean_scores[-1]:.1f}")

# Cleanup
torch.save(agent.policy_net.state_dict(), 'weight file for DQN/snake_dqn.pth')
params = agent.policy_net.state_dict()

# Note Csv files of the weights are large
with open("csv files/DQN-Weights.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    for key, weight in params.items():
        writer.writerow([key])
        flat_weights = weight.flatten().tolist()
        writer.writerow(flat_weights)  # Weights in one row
        writer.writerow([])
    print("DQN-Weight saved as csv!")

print("Saved dqn.pth")
pygame.quit()