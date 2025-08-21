# play_snake.py
import torch
import pygame
import sys
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RL_Agent_with_DQN import DQNAgent, SnakeGame, WIDTH, HEIGHT, device
from ham_cycle import prim_maze_generator, draw_cycle

from collections import deque



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

def get_neighbors(pos, grid):
    """
    Return all valid neighbors of `pos` in the grid that are walkable.
    pos: (row, col) tuple
    grid: 2D list or dict representing free cells (1 = free, 0 = blocked)
    """
    neighbors = []
    rows = len(grid)
    cols = len(grid[0])
    row, col = pos

    # Up
    if row > 0 and grid[row-1][col]:
        neighbors.append((row-1, col))
    # Down
    if row < rows-1 and grid[row+1][col]:
        neighbors.append((row+1, col))
    # Left
    if col > 0 and grid[row][col-1]:
        neighbors.append((row, col-1))
    # Right
    if col < cols-1 and grid[row][col+1]:
        neighbors.append((row, col+1))

    return neighbors


def is_tail_reachable(grid, head_pos, snake_body):
    """
    Check if the snake's head can still reach its tail.
    grid: 2D list (1=free, 0=blocked)
    head_pos: (row, col) of the new head
    snake_body: list of (row, col) positions of the snake's body (head first)
    """
    if not snake_body:
        return True

    tail_pos = snake_body[-1]

    # Make a copy of the grid so we can "free" the tail
    temp_grid = [row[:] for row in grid]
    # Free the tail cell, since it moves away next step
    tr, tc = tail_pos
    temp_grid[tr][tc] = 1

    visited = set()
    queue = deque([head_pos])

    while queue:
        r, c = queue.popleft()
        if (r, c) == tail_pos:
            return True
        for nr, nc in get_neighbors((r, c), temp_grid):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))

    return False
def find_safe_path(grid, snake_head, fruit_pos, snake_body):
    # grid: 2D list or dict of cells
    # snake_head: current head pos
    # fruit_pos: target fruit
    # snake_body: occupied positions
    # returns a list of grid positions
    from collections import deque

    visited = set(snake_body)
    queue = deque([(snake_head, [])])

    while queue:
        current, path = queue.popleft()
        if current == fruit_pos:
            # check if path leaves space (tail reachable, or cycle safe)
            if is_tail_reachable(grid, current, snake_body):
                return path + [current]
            else:
                continue  # keep searching for another safe path
        for neighbor in get_neighbors(current, grid):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [current]))
    return None  # no path found

# Setup
def should_fallback(snake, game):
    # Fallback if snake length is 75% of total grid cells
    short_threshold = (game.width // BLOCK_SIZE) * (game.height // BLOCK_SIZE) * 0.03
    long_threshold = (game.width // BLOCK_SIZE) * (game.height // BLOCK_SIZE) * 0.08
    return short_threshold, long_threshold


def rotate_cycle(cycle, head_pos):
    if head_pos in cycle:
        idx = cycle.index(head_pos)
        return cycle[idx:] + cycle[:idx]
    else:
        raise ValueError("Head position not found in cycle")

def convert_next_cell_to_action(next_cell, snake_head_pos):
    """For BFS"""
    if next_cell[0] < snake_head_pos[0]:
        action = 0  # up
    elif next_cell[0] > snake_head_pos[0]:
        action = 1  # down
    elif next_cell[1] < snake_head_pos[1]:
        action = 2  # left
    else:
        action = 3  # right
    return action

def convert_next_cell_to_ham_action(next_cell, snake_head_pos):
    if next_cell[0] > snake_head_pos[0]:
        action = 3  # right
    elif next_cell[0] < snake_head_pos[0]:
        action = 2  # left
    elif next_cell[1] > snake_head_pos[1]:
        action = 1  # down
    else:
        action = 0  # up
    return action

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
            clock.tick(60)

        print(f"Episode {episode + 1}: Total_Reward = {Total_reward:.2f}")
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    play_game(cycle)
