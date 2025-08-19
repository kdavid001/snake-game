# play_snake.py
import torch
import pygame
import sys
from RL_Agent_with_DQN import DQNAgent, SnakeGame, WIDTH, HEIGHT, device
# from Hamiltonian_cycle import generate_hamiltonian_cycle, get_cycle_action

# Setup
agent = DQNAgent()
agent.policy_net.load_state_dict(torch.load("Current DQN WEIGHTS/Best_current_weight.pth"))
agent.policy_net.to(device)
agent.policy_net.eval()
agent.epsilon = 0.0  # Greedy play

game = SnakeGame(width=WIDTH, height=HEIGHT, mode="rl")
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

BLOCK_SIZE = 20


# ham_cycle = generate_hamiltonian_cycle(game.width, game.height)

# print(type(len(game.snake.body)))
# def should_fallback(snake, game):
#     # Fallback if snake length is 75% of total grid cells
#     threshold = (game.width * game.height) * 0.0
#     return len(snake.body) >= threshold
#

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
            # snake_head_pos = (
            #     game.snake.body[0].x // BLOCK_SIZE,
            #     game.snake.body[0].y // BLOCK_SIZE
            # )
            # if should_fallback(game.snake, game):
            #     action = get_cycle_action(snake_head_pos, ham_cycle, game.width, game.height)
            # else:
            action = agent.act(current_state)
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
