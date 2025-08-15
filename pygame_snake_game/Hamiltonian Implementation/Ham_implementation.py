# play_snake.py
import torch
import pygame
import sys
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RL_Agent_with_DQN import DQNAgent, SnakeGame, WIDTH, HEIGHT, device
from ham_cycle import prim_maze_generator, draw_cycle

agent = DQNAgent()
agent.policy_net.load_state_dict(torch.load("../Current DQN WEIGHTS/Best_current_weight.pth"))
agent.policy_net.to(device)
agent.policy_net.eval()
agent.epsilon = 0.0  # Greedy play

game = SnakeGame(width=WIDTH, height=HEIGHT, mode = "cycle")
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
BLOCK_SIZE = 20

print(f"Games width and Height{game.width}, {game.height}")

rows = game.height // BLOCK_SIZE - 15
cols = game.width // BLOCK_SIZE - 20
cycle = prim_maze_generator(rows, cols)
print("Drawing maze Cycle............")
draw_cycle(cycle,game.height // BLOCK_SIZE, game.width // BLOCK_SIZE)


# Setup
def should_fallback(snake, game):
    # Fallback if snake length is 75% of total grid cells
    threshold = (game.width // BLOCK_SIZE) * (game.height // BLOCK_SIZE) * 2.0
    return len(snake.body) >= threshold


def rotate_cycle(cycle, head_pos):
    if head_pos in cycle:
        idx = cycle.index(head_pos)
        return cycle[idx:] + cycle[:idx]
    else:
        raise ValueError("Head position not found in cycle")


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
            print(f"Current_head_pos{snake_head_pos}")
            # if should_fallback(game.snake, game):
            #     # action = get_cycle_action(snake_head_pos, ham_cycle)
            #     action =get_cycle_action(snake_head_pos, cycle=cycle)
            #     print(f"{action} -> {snake_head_pos}")
            # else:
            #     action = agent.act(current_state)
            cycle_rotated = rotate_cycle(cycle, snake_head_pos)
            next_cell = cycle_rotated[1]

            # Convert to action (assuming: 0=up, 1=down, 2=left, 3=right)
            if next_cell[0] > snake_head_pos[0]:
                action = 3  # right
            elif next_cell[0] < snake_head_pos[0]:
                action = 2  # left
            elif next_cell[1] > snake_head_pos[1]:
                action = 1  # down
            else:
                action = 0  # up

            # debug: show a short slice of the rotated cycle so we can verify ordering
            print("Rotated cycle (first 6):", cycle_rotated[:6])
            print(f"Action direction: {action}  -> head: {snake_head_pos} next: {next_cell}")
            next_state, reward, done = game.step(action)
            Total_reward += reward

            game.render(screen, clock.get_fps())
            pygame.display.flip()
            clock.tick(1000)

        print(f"Episode {episode + 1}: Total_Reward = {Total_reward:.2f}")
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    play_game(cycle)
