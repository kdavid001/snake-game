""" Full Ham_implementation with a little BFS """
import torch
import pygame
import sys
import sys, os

sys.path.append(os.path.abspath("../RL_agents_Training"))
from RL_Agent_with_DDQN import DQNAgent, SnakeGame, WIDTH, HEIGHT, device
sys.path.append(os.path.abspath("../Hamiltonian_Implementation"))
from ham_cycle import prim_maze_generator, draw_cycle, find_safe_path, rotate_cycle

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

rows = game.height // BLOCK_SIZE - 15
cols = game.width // BLOCK_SIZE - 20
cycle = prim_maze_generator(rows, cols)
print("Drawing maze Cycle............")
draw_cycle(cycle, game.height // BLOCK_SIZE, game.width // BLOCK_SIZE)


# Setup
def should_fallback(snake, game):
    # Fallback if snake length is 75% of total grid cells
    threshold = (game.width // BLOCK_SIZE) * (game.height // BLOCK_SIZE) * 0.0
    game.mode = "cycle"
    return len(snake.body) >= threshold

def play_game(cycle):
    for episode in range(10):
        state = game.reset()
        current_state = agent.get_state(state).to(device)
        done = False
        Total_reward = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            snake_head_pos = (
                game.snake.body[0].x // BLOCK_SIZE,
                game.snake.body[0].y // BLOCK_SIZE
            )
            # print(f"Current_head_pos{snake_head_pos}")
            if should_fallback(game.snake, game):
                cycle_rotated = rotate_cycle(cycle, snake_head_pos)
                next_cell = cycle_rotated[1]
                if next_cell[0] > snake_head_pos[0]:
                    action = 3  # right
                elif next_cell[0] < snake_head_pos[0]:
                    action = 2  # left
                elif next_cell[1] > snake_head_pos[1]:
                    action = 1  # down
                else:
                    action = 0  # up
            else:
                action = agent.act(current_state)

            # debug: show a short slice of the rotated cycle so we can verify ordering
            # print("Rotated cycle (first 6):", cycle_rotated[:6])
            # print(f"Action direction: {action}  -> head: {snake_head_pos} next: {next_cell}")
            next_state, reward, done = game.step(action)
            current_state = agent.get_state(next_state).to(device)
            Total_reward += reward

            game.render(screen, clock.get_fps())
            pygame.display.flip()
            clock.tick(60)

        print(f"Episode {episode + 1}: Total_Reward = {Total_reward:.2f}")
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    play_game(cycle)
