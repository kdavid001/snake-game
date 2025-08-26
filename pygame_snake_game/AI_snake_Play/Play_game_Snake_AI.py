"""Snake Playing on pure AI - DDQN"""
import os
import torch
import pygame
import sys

sys.path.append(os.path.abspath("../RL_agents_Training"))
from RL_Agent_with_DDQN import DQNAgent, SnakeGame, WIDTH, HEIGHT, device


# Setup
agent = DQNAgent()
agent.policy_net.load_state_dict(torch.load("../WEIGHTS/Current DQN WEIGHTS/Best_current_weight.pth"))
agent.policy_net.to(device)
agent.policy_net.eval()
agent.epsilon = 0.0  # Greedy play

game = SnakeGame(width=WIDTH, height=HEIGHT, mode="rl")
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

BLOCK_SIZE = 20

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
            action = agent.act(current_state)
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
    play_game()
