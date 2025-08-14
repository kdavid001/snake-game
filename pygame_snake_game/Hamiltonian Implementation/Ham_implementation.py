# play_snake.py
import torch
import pygame
import sys
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RL_Agent_with_DQN import DQNAgent, SnakeGame, WIDTH, HEIGHT, device
from nan import get_cycle_action, create_cycle


agent = DQNAgent()
agent.policy_net.load_state_dict(torch.load("../Current DQN WEIGHTS/Best_current_weight.pth"))
agent.policy_net.to(device)
agent.policy_net.eval()
agent.epsilon = 0.0  # Greedy play

game = SnakeGame(width=WIDTH, height=HEIGHT)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
BLOCK_SIZE = 20

print(f"Games width and Height{game.width}, {game.height}")
precomputed_cycle = create_cycle(width= game.width, height=game.height)
print(precomputed_cycle)


# Setup

def should_fallback(snake, game):
    # Fallback if snake length is 75% of total grid cells
    threshold = (game.width // BLOCK_SIZE) * (game.height // BLOCK_SIZE) * 0.0
    return len(snake.body) >= threshold

def rotate_cycle(cycle, head_pos):
    if head_pos in cycle:
        idx = cycle.index(head_pos)
        return cycle[idx:] + cycle[:idx]
    else:
        raise ValueError("Head position not found in cycle")

def play_game():
    global game
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
            print(game.snake.body[0].x, game.snake.body[0].y)
            print(f"Current_head_pos{snake_head_pos}")
            # if should_fallback(game.snake, game):
            #     # action = get_cycle_action(snake_head_pos, ham_cycle)
            #     action =get_cycle_action(snake_head_pos, cycle=cycle)
            #     print(f"{action} -> {snake_head_pos}")
            # else:
            #     action = agent.act(current_state)
            cycle = precomputed_cycle
            cycle = rotate_cycle(cycle, snake_head_pos)
            # debug: show a short slice of the rotated cycle so we can verify ordering
            print("Rotated cycle (first 6):", cycle[:6])
            action = get_cycle_action(snake_head_pos, cycle, game.width, game.height)
            print(f"Action direction: {action}  -> head: {snake_head_pos} next: {cycle[1]}")
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
    play_game()
