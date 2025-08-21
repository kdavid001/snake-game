import os
import sys
import numpy as np
import random
from matplotlib import pyplot as plt
import pygame

sys.path.append(os.path.abspath("../game_attributes"))
from snake_game import SnakeGame
from scoreboard import Scoreboard

# Constants
width = 400
height = 400

game = SnakeGame(width, height, mode="rl")
action = ['up', 'down', 'left', 'right']
action_idx = {a: i for i, a in enumerate(action)}

# Expanded Q-table to include food position
try:
    Q = np.load("../WEIGHTS/Current_q_TABLE/Q_table_for_Rl_model.npy")
    print("Q-table loaded.")
except FileNotFoundError:
    Q = np.ones((
        width // game.block_size,
        height // game.block_size,
        width // game.block_size,
        height // game.block_size,
        len(action)
    ), dtype=float) * (1.0 / len(action))
    print("Q-table initialized.")

scoreboard = Scoreboard(width, height)
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Plot setup
plt.ion()
episode_rewards = []
fig, ax = plt.subplots()
line, = ax.plot([], [], label="Episode Reward", color='blue')
ax.set_xlim(0, 10)
ax.set_ylim(-500, 500)
ax.set_ylabel("Total Reward")
ax.set_title("Episode Rewards Over Time")
ax.legend()

# Hyperparameters
epochs = 1000
epsilon = 1.0
epsilon_decay = 0.999
alpha = 0.1
alpha_decay = 0.999
epsilon_min = 0.01
gamma = 0.9

scores = []
mean_scores = []
episodes = []
best_mean_score = float("-inf")


# Game Loop
def train(epochs):
    global alpha, epsilon, best_mean_score
    for episode in range(epochs):
        state = game.reset()
        total_reward = 0
        done = False

        x, y = state['snake_head']
        food_x, food_y = state['food']
        current_score = state['score']
        high_score = state['highscore']
        x_idx = x // game.block_size
        y_idx = y // game.block_size
        fx_idx = food_x // game.block_size
        fy_idx = food_y // game.block_size

        scoreboard.reset()
        scoreboard.update(screen, clock.get_fps())

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            if (0 <= x_idx < width // game.block_size and
                    0 <= y_idx < height // game.block_size and
                    0 <= fx_idx < width // game.block_size and
                    0 <= fy_idx < height // game.block_size):
                if random.random() < epsilon:
                    select_action = random.choice(action)
                else:
                    select_action = action[np.argmax(Q[x_idx, y_idx, fx_idx, fy_idx])]
            else:
                select_action = random.choice(action)

            next_state, reward, done = game.step(action_idx[select_action])

            new_x, new_y = next_state['snake_head']
            new_food_x, new_food_y = next_state['food']
            new_x_idx = new_x // game.block_size
            new_y_idx = new_y // game.block_size
            new_fx_idx = new_food_x // game.block_size
            new_fy_idx = new_food_y // game.block_size

            # This section is to make sure the Q-values are being updated if both the current and next state are safe to
            # index
            if (0 <= x_idx < width // game.block_size and
                    0 <= y_idx < height // game.block_size and
                    0 <= fx_idx < width // game.block_size and
                    0 <= fy_idx < height // game.block_size and
                    0 <= new_x_idx < width // game.block_size and
                    0 <= new_y_idx < height // game.block_size and
                    0 <= new_fx_idx < width // game.block_size and
                    0 <= new_fy_idx < height // game.block_size):
                # Bellman's Equation for discrete values
                current_q = Q[x_idx, y_idx, fx_idx, fy_idx, action_idx[select_action]]
                max_future_q = np.max(Q[new_x_idx, new_y_idx, new_fx_idx, new_fy_idx, :])
                new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)
                Q[x_idx, y_idx, fx_idx, fy_idx, action_idx[select_action]] = new_q

            # Updates indices for next iteration
            x_idx, y_idx = new_x_idx, new_y_idx
            fx_idx, fy_idx = new_fx_idx, new_fy_idx
            total_reward += reward

            # Renders game
            game.render(screen, clock.get_fps())
            pygame.display.flip()
            clock.tick(120)

            if done:
                epochs += 1
                episode_rewards.append(total_reward)

            #     if epochs % 10 == 0:
            #         x_vals = np.arange(len(episode_rewards))
            #         y_vals = episode_rewards
            #         line.set_data(x_vals, y_vals)
            #
            #         ax.set_xlim(0, max(10, len(episode_rewards)))
            #         ax.set_ylim(min(y_vals) - 50, max(y_vals) + 50)  # Dynamic y-limits
            #
            #         ax.relim()
            #         ax.autoscale_view()
            #         fig.canvas.draw()
            #         fig.canvas.flush_events()
        scores.append(total_reward)
        mean_score = round(np.mean(scores[-100:]), 3)
        mean_scores.append(mean_score)
        episodes.append(epochs)

        # Save best model
        if mean_score > best_mean_score:
            best_mean_score = mean_score
            np.save("../WEIGHTS/Current_q_TABLE/Q_table_for_Rl_model.npy", Q)
            print("Q-table saved!")

        scoreboard.reset()
        scoreboard.update(screen, clock.get_fps())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        alpha = max(0.01, alpha * alpha_decay)

        print(f"Episode {episode} - Reward: {total_reward} | mean:{mean_score} |Current Score: {current_score} | High Score: {high_score}| "
                  f" - ε: {epsilon:.3f} - α: {alpha:.3f}")


# Save the Q-table


if __name__ == "__main__":
    train(epochs)
    pygame.quit()
