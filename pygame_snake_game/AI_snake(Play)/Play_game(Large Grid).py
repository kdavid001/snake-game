"""Smame but for Bigger Grid 800 X 600 """
"""Still working on this for Improvement """

import torch
import pygame
import sys
import sys, os
from collections import deque

sys.path.append(os.path.abspath("../RL_agents(Training)"))
from RL_Agent_with_DDQN import DQNAgent, SnakeGame, WIDTH, HEIGHT, device

sys.path.append(os.path.abspath("../Hamiltonian_Implementation"))
from ham_cycle import (prim_maze_generator, draw_cycle, find_safe_path,
                       convert_next_cell_to_ham_action, convert_next_cell_to_action, rotate_cycle, is_tail_reachable)

agent = DQNAgent()
agent.policy_net.load_state_dict(torch.load("../WEIGHTS/Current DQN WEIGHTS/Weight_wn_Reward_sys.pth"))
agent.policy_net.to(device)
agent.policy_net.eval()
agent.epsilon = 0.0  # Greedy play

game = SnakeGame(width=WIDTH, height=HEIGHT)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
BLOCK_SIZE = 20

print(f"Games width and Height{game.width}, {game.height}")

rows = game.height // BLOCK_SIZE - ((game.height // BLOCK_SIZE) // 2)
cols = game.width // BLOCK_SIZE - ((game.width // BLOCK_SIZE) // 2)
cycle = prim_maze_generator(rows, cols)
print("Drawing maze Cycle............")
draw_cycle(cycle, game.height // BLOCK_SIZE, game.width // BLOCK_SIZE)

# Setup
def should_fallback(snake, game):
    short_threshold = (game.width // BLOCK_SIZE) * (game.height // BLOCK_SIZE) * 0.03
    long_threshold = (game.width // BLOCK_SIZE) * (game.height // BLOCK_SIZE) * 0.04
    return short_threshold, long_threshold


def play_game(cycle):
    for episode in range(20):
        state = game.reset()
        current_state = agent.get_state(state).to(device)
        done = False
        Total_reward = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()



            short_threshold, long_threshold = should_fallback(cycle, game)
            if len(game.snake.body) < short_threshold:
                """ Use DQN agent """
                game.mode = "rl"  # rl -> Reinforcement learning agent
                action = agent.act(current_state)

            elif len(game.snake.body) < long_threshold:
                """ Use BFS shortcut """
                game.mode = "cycle"
                # Note: here BFS expects (x,y) unlike Ham_cycle
                snake_head_pos = (
                    game.snake.body[0].y // BLOCK_SIZE,  # row
                    game.snake.body[0].x // BLOCK_SIZE  # col
                )
                fruit_pos = (
                    game.food.rect.y // BLOCK_SIZE,  # row
                    game.food.rect.x // BLOCK_SIZE  # col
                )
                snake_body_cells = [(seg.y // BLOCK_SIZE, seg.x // BLOCK_SIZE) for seg in game.snake.body]

                grid = [[1] * (game.width // BLOCK_SIZE) for _ in range(game.height // BLOCK_SIZE)]
                for r, c in snake_body_cells:
                    grid[r][c] = 0
                safe_path = find_safe_path(grid, snake_head_pos, fruit_pos, snake_body_cells)

                if safe_path:
                    next_cell = safe_path[1]  # the next cell to move to
                    action = convert_next_cell_to_action(next_cell, snake_head_pos)
            # elif len(game.snake.body) > long_threshold:
            else:
                """ Use Hamiltonian cycle """
                snake_head_pos = (
                    game.snake.body[0].x // BLOCK_SIZE,
                    game.snake.body[0].y // BLOCK_SIZE
                )
                game.mode = "cycle"
                cycle_rotated = rotate_cycle(cycle, snake_head_pos)
                next_cell = cycle_rotated[1]
                action = convert_next_cell_to_ham_action(next_cell, snake_head_pos)

            next_state, reward, done = game.step(action)
            current_state = agent.get_state(next_state).to(device)
            Total_reward += reward

            game.render(screen, clock.get_fps())
            pygame.display.flip()
            clock.tick(1000)

        print(f"Episode {episode + 1}: Total_Reward = {Total_reward:.2f}")
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    play_game(cycle)
