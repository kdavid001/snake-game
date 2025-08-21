"""
Taking the Ham_cycle apart to get a proper understanding
"""
from random import randint
import matplotlib.pyplot as plt
from collections import deque


# Generates a path composed of coordinates for the snake to travel along
def path_generator(graph, cells):
    # The starting position for the path is at cell (0, 0)
    path = [(0, 0)]

    previous_cell = path[0]
    previous_direction = None

    # Generates a path that is a hamiltonian cycle by following a set of general laws
    # 1. If the right cell is available, travel to the right
    # 2. If the cell underneath is available, travel down
    # 3. If the left cell is available, travel left
    # 4. If the cell above is available, travel up
    # 5. The current direction cannot oppose the previous direction (e.g. left --> right)
    while len(path) != cells:

        if previous_cell in graph and (previous_cell[0] + 1, previous_cell[1]) in graph[previous_cell] \
                and previous_direction != 'left':
            path.append((previous_cell[0] + 1, previous_cell[1]))
            previous_cell = (previous_cell[0] + 1, previous_cell[1])
            previous_direction = 'right'
        elif previous_cell in graph and (previous_cell[0], previous_cell[1] + 1) in graph[previous_cell] \
                and previous_direction != 'up':
            path.append((previous_cell[0], previous_cell[1] + 1))
            previous_cell = (previous_cell[0], previous_cell[1] + 1)
            previous_direction = 'down'
        elif (previous_cell[0] - 1, previous_cell[1]) in graph \
                and previous_cell in graph[previous_cell[0] - 1, previous_cell[1]] and previous_direction != 'right':
            path.append((previous_cell[0] - 1, previous_cell[1]))
            previous_cell = (previous_cell[0] - 1, previous_cell[1])
            previous_direction = 'left'
        else:
            path.append((previous_cell[0], previous_cell[1] - 1))
            previous_cell = (previous_cell[0], previous_cell[1] - 1)
            previous_direction = 'up'

    # Returns the coordinates of the hamiltonian cycle path
    return path


def prim_maze_generator(grid_rows, grid_columns):
    directions = dict()
    vertices = grid_rows * grid_columns

    # Creates keys for the directions dictionary
    # Note that the maze has half the width and length of the grid for the hamiltonian cycle
    for i in range(grid_rows):
        for j in range(grid_columns):
            directions[j, i] = []

    # The initial cell for maze generation is chosen randomly
    x = randint(0, grid_columns - 1)
    y = randint(0, grid_rows - 1)
    initial_cell = (x, y)

    current_cell = initial_cell

    # Stores all cells that have been visited
    visited = [initial_cell]

    # Contains all neighbouring cells to cells that have been visited
    adjacent_cells = set()

    # Generates walls in grid randomly to create a randomized maze
    while len(visited) != vertices:

        # Stores the position of the current cell in the grid
        x_position = current_cell[0]
        y_position = current_cell[1]

        # Finds adjacent cells when the current cell does not lie on the edge of the grid
        if x_position != 0 and y_position != 0 and x_position != grid_columns - 1 and y_position != grid_rows - 1:
            adjacent_cells.add((x_position, y_position - 1))
            adjacent_cells.add((x_position, y_position + 1))
            adjacent_cells.add((x_position - 1, y_position))
            adjacent_cells.add((x_position + 1, y_position))

        # Finds adjacent cells when the current cell lies in the left top corner of the grid
        elif x_position == 0 and y_position == 0:
            adjacent_cells.add((x_position + 1, y_position))
            adjacent_cells.add((x_position, y_position + 1))

        # Finds adjacent cells when the current cell lies in the bottom left corner of the grid
        elif x_position == 0 and y_position == grid_rows - 1:
            adjacent_cells.add((x_position, y_position - 1))
            adjacent_cells.add((x_position + 1, y_position))

        # Finds adjacent cells when the current cell lies in the left column of the grid
        elif x_position == 0:
            adjacent_cells.add((x_position, y_position - 1))
            adjacent_cells.add((x_position, y_position + 1))
            adjacent_cells.add((x_position + 1, y_position))

        # Finds adjacent cells when the current cell lies in the top right corner of the grid
        elif x_position == grid_columns - 1 and y_position == 0:
            adjacent_cells.add((x_position, y_position + 1))
            adjacent_cells.add((x_position - 1, y_position))

        # Finds adjacent cells when the current cell lies in the bottom right corner of the grid
        elif x_position == grid_columns - 1 and y_position == grid_rows - 1:
            adjacent_cells.add((x_position, y_position - 1))
            adjacent_cells.add((x_position - 1, y_position))

        # Finds adjacent cells when the current cell lies in the right column of the grid
        elif x_position == grid_columns - 1:
            adjacent_cells.add((x_position, y_position - 1))
            adjacent_cells.add((x_position, y_position + 1))
            adjacent_cells.add((x_position - 1, y_position))

        # Finds adjacent cells when the current cell lies in the top row of the grid
        elif y_position == 0:
            adjacent_cells.add((x_position, y_position + 1))
            adjacent_cells.add((x_position - 1, y_position))
            adjacent_cells.add((x_position + 1, y_position))

        # Finds adjacent cells when the current cell lies in the bottom row of the grid
        else:
            adjacent_cells.add((x_position, y_position - 1))
            adjacent_cells.add((x_position + 1, y_position))
            adjacent_cells.add((x_position - 1, y_position))

        # Generates a wall between two cells in the grid
        while current_cell:

            current_cell = (adjacent_cells.pop())

            # The neighbouring cell is disregarded if it is already a wall in the maze
            if current_cell not in visited:

                # The neighbouring cell is now classified as having been visited
                visited.append(current_cell)
                x = current_cell[0]
                y = current_cell[1]

                # To generate a wall, a cell adjacent to the current cell must already have been visited
                # The direction of the wall between cells is stored
                # The process is simplified by only considering a wall to be to the right or down
                if (x + 1, y) in visited:
                    directions[x, y] += ['right']
                elif (x - 1, y) in visited:
                    directions[x - 1, y] += ['right']
                elif (x, y + 1) in visited:
                    directions[x, y] += ['down']
                elif (x, y - 1) in visited:
                    directions[x, y - 1] += ['down']

                break

    # Provides the hamiltonian cycle generating algorithm with the direction of the walls to avoid
    return hamiltonian_cycle(grid_rows, grid_columns, directions)


def hamiltonian_cycle(grid_rows, grid_columns, orientation):
    # The path for the snake is stored in a dictionary
    # The keys are the (x, y) positions in the grid
    # The values are the adjacent (x, y) positions that the snake can travel towards
    hamiltonian_graph = dict()

    # Uses the coordinates of the walls to generate available adjacent cells for each cell
    # Simplified by only considering the right and down directions
    for i in range(grid_rows):
        for j in range(grid_columns):

            # Finds available adjacent cells if current cell does not lie on an edge of the grid
            if j != grid_columns - 1 and i != grid_rows - 1 and j != 0 and i != 0:
                if 'right' in orientation[j, i]:
                    hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 2, i * 2)]
                    hamiltonian_graph[j * 2 + 1, i * 2 + 1] = [(j * 2 + 2, i * 2 + 1)]
                else:
                    hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 1, i * 2 + 1)]
                if 'down' in orientation[j, i]:
                    hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2, i * 2 + 2)]
                    if (j * 2 + 1, i * 2 + 1) in hamiltonian_graph:
                        hamiltonian_graph[j * 2 + 1, i * 2 + 1] += [(j * 2 + 1, i * 2 + 2)]
                    else:
                        hamiltonian_graph[j * 2 + 1, i * 2 + 1] = [(j * 2 + 1, i * 2 + 2)]
                else:
                    hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2 + 1, i * 2 + 1)]
                if 'down' not in orientation[j, i - 1]:
                    hamiltonian_graph[j * 2, i * 2] = [(j * 2 + 1, i * 2)]
                if 'right' not in orientation[j - 1, i]:
                    if (j * 2, i * 2) in hamiltonian_graph:
                        hamiltonian_graph[j * 2, i * 2] += [(j * 2, i * 2 + 1)]
                    else:
                        hamiltonian_graph[j * 2, i * 2] = [(j * 2, i * 2 + 1)]

            # Finds available adjacent cells if current cell is in the bottom right corner
            elif j == grid_columns - 1 and i == grid_rows - 1:
                hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2 + 1, i * 2 + 1)]
                hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 1, i * 2 + 1)]
                if 'down' not in orientation[j, i - 1]:
                    hamiltonian_graph[j * 2, i * 2] = [(j * 2 + 1, i * 2)]
                elif 'right' not in orientation[j - 1, i]:
                    hamiltonian_graph[j * 2, i * 2] = [(j * 2, i * 2 + 1)]

            # Finds available adjacent cells if current cell is in the top right corner
            elif j == grid_columns - 1 and i == 0:
                hamiltonian_graph[j * 2, i * 2] = [(j * 2 + 1, i * 2)]
                hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 1, i * 2 + 1)]
                if 'down' in orientation[j, i]:
                    hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2, i * 2 + 2)]
                    hamiltonian_graph[j * 2 + 1, i * 2 + 1] = [(j * 2 + 1, i * 2 + 2)]
                else:
                    hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2 + 1, i * 2 + 1)]
                if 'right' not in orientation[j - 1, i]:
                    hamiltonian_graph[j * 2, i * 2] += [(j * 2, i * 2 + 1)]

            # Finds available adjacent cells if current cell is in the right column
            elif j == grid_columns - 1:
                hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 1, i * 2 + 1)]
                if 'down' in orientation[j, i]:
                    hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2, i * 2 + 2)]
                    hamiltonian_graph[j * 2 + 1, i * 2 + 1] = [(j * 2 + 1, i * 2 + 2)]
                else:
                    hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2 + 1, i * 2 + 1)]
                if 'down' not in orientation[j, i - 1]:
                    hamiltonian_graph[j * 2, i * 2] = [(j * 2 + 1, i * 2)]
                if 'right' not in orientation[j - 1, i]:
                    if (j * 2, i * 2) in hamiltonian_graph:
                        hamiltonian_graph[j * 2, i * 2] += [(j * 2, i * 2 + 1)]
                    else:
                        hamiltonian_graph[j * 2, i * 2] = [(j * 2, i * 2 + 1)]

            # Finds available adjacent cells if current cell is in the top left corner
            elif j == 0 and i == 0:
                hamiltonian_graph[j * 2, i * 2] = [(j * 2 + 1, i * 2)]
                hamiltonian_graph[j * 2, i * 2] += [(j * 2, i * 2 + 1)]
                if 'right' in orientation[j, i]:
                    hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 2, i * 2)]
                    hamiltonian_graph[j * 2 + 1, i * 2 + 1] = [(j * 2 + 2, i * 2 + 1)]
                else:
                    hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 1, i * 2 + 1)]
                if 'down' in orientation[j, i]:
                    hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2, i * 2 + 2)]
                    if (j * 2 + 1, i * 2 + 1) in hamiltonian_graph:
                        hamiltonian_graph[j * 2 + 1, i * 2 + 1] += [(j * 2 + 1, i * 2 + 2)]
                    else:
                        hamiltonian_graph[j * 2 + 1, i * 2 + 1] = [(j * 2 + 1, i * 2 + 2)]
                else:
                    hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2 + 1, i * 2 + 1)]

            # Finds available adjacent cells if current cell is in the bottom left corner
            elif j == 0 and i == grid_rows - 1:
                hamiltonian_graph[j * 2, i * 2] = [(j * 2, i * 2 + 1)]
                hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2 + 1, i * 2 + 1)]
                if 'right' in orientation[j, i]:
                    hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 2, i * 2)]
                    hamiltonian_graph[j * 2 + 1, i * 2 + 1] = [(j * 2 + 2, i * 2 + 1)]
                else:
                    hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 1, i * 2 + 1)]
                if 'down' not in orientation[j, i - 1]:
                    hamiltonian_graph[j * 2, i * 2] += [(j * 2 + 1, i * 2)]

            # Finds available adjacent cells if current cell is in the left corner
            elif j == 0:
                hamiltonian_graph[j * 2, i * 2] = [(j * 2, i * 2 + 1)]
                if 'right' in orientation[j, i]:
                    hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 2, i * 2)]
                    hamiltonian_graph[j * 2 + 1, i * 2 + 1] = [(j * 2 + 2, i * 2 + 1)]
                else:
                    hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 1, i * 2 + 1)]
                if 'down' in orientation[j, i]:
                    hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2, i * 2 + 2)]
                    if (j * 2 + 1, i * 2 + 1) in hamiltonian_graph:
                        hamiltonian_graph[j * 2 + 1, i * 2 + 1] += [(j * 2 + 1, i * 2 + 2)]
                    else:
                        hamiltonian_graph[j * 2 + 1, i * 2 + 1] = [(j * 2 + 1, i * 2 + 2)]
                else:
                    hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2 + 1, i * 2 + 1)]
                if 'down' not in orientation[j, i - 1]:
                    hamiltonian_graph[j * 2, i * 2] += [(j * 2 + 1, i * 2)]

            # Finds available adjacent cells if current cell is in the top row
            elif i == 0:
                hamiltonian_graph[j * 2, i * 2] = [(j * 2 + 1, i * 2)]
                if 'right' in orientation[j, i]:
                    hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 2, i * 2)]
                    hamiltonian_graph[j * 2 + 1, i * 2 + 1] = [(j * 2 + 2, i * 2 + 1)]
                else:
                    hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 1, i * 2 + 1)]
                if 'down' in orientation[j, i]:
                    hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2, i * 2 + 2)]
                    if (j * 2 + 1, i * 2 + 1) in hamiltonian_graph:
                        hamiltonian_graph[j * 2 + 1, i * 2 + 1] += [(j * 2 + 1, i * 2 + 2)]
                    else:
                        hamiltonian_graph[j * 2 + 1, i * 2 + 1] = [(j * 2 + 1, i * 2 + 2)]
                else:
                    hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2 + 1, i * 2 + 1)]
                if 'right' not in orientation[j - 1, i]:
                    hamiltonian_graph[j * 2, i * 2] += [(j * 2, i * 2 + 1)]

            # Finds available adjacent cells if current cell is in the bottom row
            else:
                hamiltonian_graph[j * 2, i * 2 + 1] = [(j * 2 + 1, i * 2 + 1)]
                if 'right' in orientation[j, i]:
                    hamiltonian_graph[j * 2 + 1, i * 2 + 1] = [(j * 2 + 2, i * 2 + 1)]
                    hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 2, i * 2)]
                else:
                    hamiltonian_graph[j * 2 + 1, i * 2] = [(j * 2 + 1, i * 2 + 1)]
                if 'down' not in orientation[j, i - 1]:
                    hamiltonian_graph[j * 2, i * 2] = [(j * 2 + 1, i * 2)]
                if 'right' not in orientation[j - 1, i]:
                    if (j * 2, i * 2) in hamiltonian_graph:
                        hamiltonian_graph[j * 2, i * 2] += [(j * 2, i * 2 + 1)]
                    else:
                        hamiltonian_graph[j * 2, i * 2] = [(j * 2, i * 2 + 1)]

    # Provides the coordinates of available adjacent cells to generate directions for the snake's movement
    return path_generator(hamiltonian_graph, grid_rows * grid_columns * 4)


def draw_cycle(cycle, height, width):
    """Visualize the cycle"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw grid
    for x in range(width + 1):
        ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)
    for y in range(height + 1):
        ax.axhline(y, color='gray', linestyle='--', linewidth=0.5)

    # Plot path
    x, y = zip(*cycle)
    ax.plot(x, y, 'b-', alpha=0.5)
    ax.scatter(x, y, c=range(len(cycle)), cmap='viridis', s=50)

    # Mark start and end
    ax.plot(x[0], y[0], 'go', markersize=10, label='Start')
    ax.plot(x[-1], y[-1], 'ro', markersize=10, label='End')

    ax.set_aspect('equal')
    ax.invert_yaxis()
    plt.xticks(range(width))
    plt.yticks(range(height))
    plt.title(f"Random Hamiltonian Cycle ({width}x{height})")
    plt.legend()
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, label='Path Order')
    plt.show()

    # def draw_cycle(cycle, height, width):
    #     """Visualize the cycle with clear grid and cell overlay"""
    #     import matplotlib.patches as patches
    #     fig, ax = plt.subplots(figsize=(12, 9))  # larger figure for clarity
    #
    #     # Draw grid cells as rectangles
    #     for i in range(height):
    #         for j in range(width):
    #             rect = patches.Rectangle((j, i), 1, 1, linewidth=1.2, edgecolor='#888', facecolor='#f8f8f8', zorder=1)
    #             ax.add_patch(rect)
    #
    #     # Draw grid lines (lighter for debugging)
    #     for x in range(width + 1):
    #         ax.axvline(x, color='#bbb', linestyle='-', linewidth=0.7, zorder=2)
    #     for y in range(height + 1):
    #         ax.axhline(y, color='#bbb', linestyle='-', linewidth=0.7, zorder=2)
    #
    #     # Plot path line over grid
    #     x, y = zip(*cycle)
    #     ax.plot(x, y, 'b-', alpha=0.7, linewidth=2.2, zorder=3)
    #     ax.scatter(x, y, c=range(len(cycle)), cmap='viridis', s=80, zorder=4)
    #
    #     # Mark start and end clearly with edge
    #     ax.plot(x[0], y[0], 'go', markersize=16, markeredgecolor='black', markeredgewidth=2, label='Start', zorder=5)
    #     ax.plot(x[-1], y[-1], 'ro', markersize=16, markeredgecolor='black', markeredgewidth=2, label='End', zorder=5)
    #
    #     # Draw a thicker rectangle around start/end cell for clarity
    #     start_rect = patches.Rectangle((x[0]//1, y[0]//1), 1, 1, linewidth=2.5, edgecolor='green', facecolor='none', zorder=6)
    #     ax.add_patch(start_rect)
    #     end_rect = patches.Rectangle((x[-1]//1, y[-1]//1), 1, 1, linewidth=2.5, edgecolor='red', facecolor='none', zorder=6)
    #     ax.add_patch(end_rect)
    #
    #     # Set axis limits and aspect
    #     ax.set_xlim(-0.5, width + 0.5)
    #     ax.set_ylim(-0.5, height + 0.5)
    #     ax.set_aspect('equal')
    #     ax.invert_yaxis()
    #     plt.xticks(range(width))
    #     plt.yticks(range(height))
    #     plt.grid(False)
    #     plt.title(f"Hamiltonian Cycle Visualization ({width}x{height})")
    #     plt.legend()
    #     plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, label='Path Order')
    #     plt.tight_layout()
    #     plt.show()


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
    if row > 0 and grid[row - 1][col]:
        neighbors.append((row - 1, col))
    # Down
    if row < rows - 1 and grid[row + 1][col]:
        neighbors.append((row + 1, col))
    # Left
    if col > 0 and grid[row][col - 1]:
        neighbors.append((row, col - 1))
    # Right
    if col < cols - 1 and grid[row][col + 1]:
        neighbors.append((row, col + 1))

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


def rotate_cycle(cycle, head_pos):
    if head_pos in cycle:
        idx = cycle.index(head_pos)
        return cycle[idx:] + cycle[:idx]
    else:
        raise ValueError("Head position not found in cycle")


if __name__ == '__main__':
    """ Width x Height """
    cycle = prim_maze_generator(600 // 40, 800 // 40)
    draw_cycle(cycle,height=600//40, width=800//40)
    print(cycle)
    width, height = 40, 30
