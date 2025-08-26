import os
import sys

import numpy as np
import random
import pygame
import sys
import math
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../game_attributes"))
from snake_game import SnakeGame

# Game Constants
WIDTH, HEIGHT = 400, 400
BLOCK_SIZE = 20
GRID_WIDTH = WIDTH // BLOCK_SIZE
GRID_HEIGHT = HEIGHT // BLOCK_SIZE

# Initialize Game
game = SnakeGame(width=WIDTH, height=HEIGHT, mode="rl")
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Q-Learning Parameters
ACTIONS = ['up', 'down', 'left', 'right']
ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}
STATE_DIM = (GRID_WIDTH, GRID_HEIGHT, 4, 3)  # grid_x, grid_y, food_dir, danger_level
Q = np.random.uniform(-1, 1, (*STATE_DIM, len(ACTIONS)))

# Hyperparameters
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
ALPHA = 0.2
GAMMA = 0.95
EPISODES = 5000
TRAINING_MODE = True
RENDER_EVERY = 1

# Tracking
scores = []
mean_scores = []
episodes = []
best_mean_score = float("-inf")


def get_state(game_state):
    """Convert game state to Q-table indices"""
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
    danger_level = min(danger, 2)  # I originally used 3 but it didnt work so 2 is the best.

    return grid_x, grid_y, food_dir, danger_level


def choose_action(state):
    """ε-greedy action selection"""
    return random.choice(ACTIONS) if random.random() < EPSILON else ACTIONS[np.argmax(Q[state])]


# Training Loop
for episode in range(EPISODES):
    state = game.reset()
    total_reward = 0
    done = False
    current_state = get_state(state)

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                np.save("../WEIGHTS/Current_q_TABLE/snake_q_table.npy", Q)
                print("Q-table saved")
                pygame.quit()
                sys.exit()

        action = choose_action(current_state)
        next_state, reward, done = game.step(ACTION_INDEX[action])

        # Reward shaping
        prev_dist = math.dist(state['snake_head'], state['food'])
        new_dist = math.dist(next_state['snake_head'], next_state['food'])
        reward += 0.1 * (prev_dist - new_dist) / BLOCK_SIZE

        # Q-learning update
        next_state_features = get_state(next_state)
        if not done:
            Q[current_state + (ACTION_INDEX[action],)] = (1 - ALPHA) * Q[current_state + (ACTION_INDEX[action],)] + \
                                                         ALPHA * (reward + GAMMA * np.max(Q[next_state_features]))

        current_state = next_state_features
        total_reward += reward

        # Conditional rendering
        game.render(screen, clock.get_fps())
        pygame.display.flip()
        clock.tick(300)

    # Store results
    scores.append(total_reward)
    mean_score = round(np.mean(scores[-100:]), 3)
    mean_scores.append(mean_score)
    episodes.append(episode)

    # Save best model
    if mean_score > best_mean_score:
        best_mean_score = mean_score
        np.save("../WEIGHTS/Current_q_TABLE/snake_q_table.npy", Q)
        print("Q-table saved!")
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

    # TODO: Try saving this in a dictionary or file so it can be plotted.
    print(f"Ep {episode:04d} | Rewards: {total_reward:3.0f} | ε: {EPSILON:.3f} | Mean: {mean_scores[-1]:.1f}")

plt.show()
pygame.quit()
