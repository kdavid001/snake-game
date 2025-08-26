""" SG - Small Grid (400 X 400) """

import numpy as np
import random
import pygame
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

sys.path.append(os.path.abspath("../game_attributes"))
from snake_game import SnakeGame

# Game Constants
WIDTH, HEIGHT = 400, 400
BLOCK_SIZE = 20
GRID_WIDTH = WIDTH // BLOCK_SIZE
GRID_HEIGHT = HEIGHT // BLOCK_SIZE

# Neural Network Parameters
STATE_SIZE = 13
BATCH_SIZE = 128
MEMORY_SIZE = 10000
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001

# Initialize Game
game = SnakeGame(width=WIDTH, height=HEIGHT, mode="rl")
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


class DQN(nn.Module):
    """Deep Q-Network with state representation"""

    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # Feed-forward
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.fc(x)


class DQNAgent:
    def __init__(self):
        # Policy Network
        self.policy_net = DQN(STATE_SIZE, 4)  # 14 input features now
        self.target_net = DQN(STATE_SIZE, 4)
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
        body = game_state['snake_body']
        direction = game_state['direction']  # Assume direction is a tuple like (dx, dy)

        # Normalized head position
        grid_x = head[0] / WIDTH
        grid_y = head[1] / HEIGHT

        # Relative food direction (normalized)
        dx = (food[0] - head[0]) / WIDTH
        dy = (food[1] - head[1]) / HEIGHT

        # Snake direction (one-hot encoded)
        dir_up = 1 if direction == (0, -BLOCK_SIZE) else 0
        dir_right = 1 if direction == (BLOCK_SIZE, 0) else 0
        dir_down = 1 if direction == (0, BLOCK_SIZE) else 0
        dir_left = 1 if direction == (-BLOCK_SIZE, 0) else 0

        # Danger detection (binary)
        def is_danger(pos):
            x, y = pos
            return x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT or (x, y) in body

        left = (head[0] - BLOCK_SIZE, head[1])
        right = (head[0] + BLOCK_SIZE, head[1])
        up = (head[0], head[1] - BLOCK_SIZE)
        down = (head[0], head[1] + BLOCK_SIZE)

        danger_left = int(is_danger(left))
        danger_right = int(is_danger(right))
        danger_up = int(is_danger(up))
        danger_down = int(is_danger(down))

        # Snake length normalized
        max_possible_length = (GRID_WIDTH * GRID_HEIGHT)
        snake_length = len(body) // max_possible_length

        # state vector
        state = [
            grid_x, grid_y,  # head position
            dx, dy,  # relative food position
            danger_left,
            danger_right,
            danger_up,
            danger_down,
            dir_up,
            dir_right,
            dir_down,
            dir_left,
            snake_length,
            # 1.0 # constant bias term
        ]

        return torch.FloatTensor(state)

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
        # unpacks them into separate lists
        states, actions, rewards, next_states, dones = zip(*batch)

        # Converts to tensors and move to GPU
        states = torch.stack(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.stack(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
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
best_mean_score = 1700  # from Last training

# TODO: Change the file name
# Model Loading
NEW_WEIGHT_PATH = "../WEIGHTS/Current DQN WEIGHTS/400x400_state_space(environment).pth"
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

    for episode in range(2000):
        state = game.reset()
        current_state = agent.get_state(state).to(device)
        total_reward = 0
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action = agent.act(current_state)
            # print(f"Action: {action}, {type(action)}")
            next_state, reward, done = game.step(action)
            # reward = reward.to(device)
            next_state_processed = agent.get_state(next_state).to(device)

            # Store experience with negative reward for collisions
            agent.remember(current_state, action, reward, next_state_processed, done)
            agent.learn()

            current_state = next_state_processed
            total_reward += reward

            # Rendering
            game.render(screen, clock.get_fps())
            pygame.display.flip()
            clock.tick(60)  # Reduce speed for better observation

        # Episode statistics
        scores.append(total_reward)
        mean_score = np.mean(scores[-100:])
        mean_scores.append(mean_score)
        episodes.append(episode)

        # Save best model
        if mean_score > best_mean_score:
            best_mean_score = mean_score
            torch.save(agent.policy_net.state_dict(), NEW_WEIGHT_PATH)
            print("Saved new weights")

        # Plotting the Scores overtime
        # if episode % 20 == 0 and episode != 0:
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

    pygame.quit()


if __name__ == '__main__':
    train_model()
    pygame.quit()
