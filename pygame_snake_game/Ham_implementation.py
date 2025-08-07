# play_snake.py
import torch
import pygame
import sys
from RL_Agent_with_DQN import DQNAgent, SnakeGame, WIDTH, HEIGHT, device
from nan import start_cycle, get_cycle_action
from ham_cycle import HamiltonianCycle, get_cycle_action


agent = DQNAgent()
agent.policy_net.load_state_dict(torch.load("Current DQN WEIGHTS/Best_current_weight.pth"))
agent.policy_net.to(device)
agent.policy_net.eval()
agent.epsilon = 0.0  # Greedy play

game = SnakeGame(width=WIDTH, height=HEIGHT)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
BLOCK_SIZE = 20

print(game.height, game.width)
# ham_cycle = start_cycle(game.height, game.width)
ham_cycle = HamiltonianCycle(game.height, game.width,)
cycle = ham_cycle.create_cycle()
# Setup

def should_fallback(snake, game):
    # Fallback if snake length is 75% of total grid cells
    threshold = (game.width // BLOCK_SIZE) * (game.height // BLOCK_SIZE) * 0.0
    return len(snake.body) >= threshold


def play_game():
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

            if should_fallback(game.snake, game):
                # action = get_cycle_action(snake_head_pos, ham_cycle)
                action =get_cycle_action(snake_head_pos, cycle=cycle)
                print(f"{action} -> {snake_head_pos}")
            else:
                action = agent.act(current_state)
            next_state, reward, done = game.step(action)
            print(f"next_state: {next_state}")
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
