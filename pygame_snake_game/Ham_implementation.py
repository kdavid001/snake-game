# play_snake.py
import torch
import pygame
import sys
from RL_Agent_with_DQN import DQNAgent, SnakeGame, WIDTH, HEIGHT, device
from Hamiltonian_cycle import get_cycle_action, generate_random_hamiltonian

# Setup
agent = DQNAgent()
agent.policy_net.load_state_dict(torch.load("Current DQN WEIGHTS/new_dqn_weights.pth"))
agent.policy_net.to(device)
agent.policy_net.eval()
agent.epsilon = 0.0  # Greedy play

game = SnakeGame(width=WIDTH, height=HEIGHT)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()


# print(type(len(game.snake.body)))
def should_fallback(snake, game):
    # Fallback if snake length is 75% of total grid cells
    threshold = (game.width * game.height) * 0.0
    return len(snake.body) >= threshold




def play_game():
    for episode in range(10):
        BLOCK_SIZE = 20  # or 20, whatever you're using
        print(game.width, game.height)
        GRID_WIDTH = game.width // BLOCK_SIZE
        GRID_HEIGHT = game.height // BLOCK_SIZE
        print(GRID_WIDTH, GRID_HEIGHT)
        ham_cycle = generate_random_hamiltonian(GRID_WIDTH, GRID_HEIGHT)

        state = game.reset()
        # Align Hamiltonian cycle to snake's head position
        snake_head_pos = (game.snake.body[0].x // BLOCK_SIZE, game.snake.body[0].y // BLOCK_SIZE)
        if snake_head_pos in ham_cycle:
            idx = ham_cycle.index(snake_head_pos)
            ham_cycle = ham_cycle[idx:] + ham_cycle[:idx]
        else:
            pass
            # print("Snake head not in Hamiltonian cycle! This will cause fallback issues.")
        current_state = agent.get_state(state).to(device)
        done = False
        Total_reward = 0
        for i in range(len(ham_cycle)):
            curr = ham_cycle[i]
            next_ = ham_cycle[(i + 1) % len(ham_cycle)]  # wrap around for cycle
            dx = abs(curr[0] - next_[0])
            dy = abs(curr[1] - next_[1])
            if dx + dy != 1:
                print(f"Invalid move between {curr} and {next_}")
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()  # Quit immediately if window is closed

            snake_head_pos = (game.snake.body[0].x // BLOCK_SIZE,game.snake.body[0].y // BLOCK_SIZE)
            if should_fallback(game.snake, game):
                action = get_cycle_action(snake_head_pos, ham_cycle)
            else:
                action = agent.act(current_state)
            print(f"Checking if action is a tuple or a list : {type(action), action}")
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
