from prims_algorithm import PrimsAlgorithm
import time
import random


class HamiltonianCycle:
    def __init__(self, columns, rows):
        self.columns = columns
        self.rows = rows

    # def create_cycle(self, nodes, final_edges):
    #     points = []
    #
    #     for edge in final_edges:
    #         start = dict(nodes[edge['startNode']])
    #         end = dict(nodes[edge['endNode']])
    #         mid_x = (start['x'] + end['x']) // 2
    #         mid_y = (start['y'] + end['y']) // 2
    #         mid = {'x': mid_x, 'y': mid_y}
    #
    #         points.extend([start, mid, end])
    #
    #     def is_in_points(x, y):
    #         return any(pos['x'] == x and pos['y'] == y for pos in points)
    #
    #     cycle = [{'x': 0, 'y': 0}]
    #     current = cycle[0]
    #
    #     def is_in_cycle(pos):
    #         return any(p['x'] == pos['x'] and p['y'] == pos['y'] for p in cycle)
    #
    #     directions = {
    #         'up': {'x': 0, 'y': -1},
    #         'down': {'x': 0, 'y': 1},
    #         'left': {'x': -1, 'y': 0},
    #         'right': {'x': 1, 'y': 0},
    #     }
    #
    #     direction = directions['right']
    #
    #     while len(cycle) < self.columns * self.rows:
    #         x, y = current['x'], current['y']
    #         dir_x = direction['x'] + x
    #         dir_y = direction['y'] + y
    #
    #         if direction == directions['right']:
    #             if is_in_points(x + 1, y + 1) and not is_in_points(x + 1, y):
    #                 current = {'x': dir_x, 'y': dir_y}
    #             elif is_in_points(x, y + 1) and not is_in_points(x + 1, y + 1):
    #                 direction = directions['down']
    #             else:
    #                 direction = directions['up']
    #
    #         elif direction == directions['down']:
    #             if is_in_points(x, y + 1) and not is_in_points(x + 1, y + 1):
    #                 current = {'x': dir_x, 'y': dir_y}
    #             elif is_in_points(x, y + 1) and is_in_points(x + 1, y + 1):
    #                 direction = directions['right']
    #             else:
    #                 direction = directions['left']
    #
    #         elif direction == directions['left']:
    #             if is_in_points(x, y) and not is_in_points(x, y + 1):
    #                 current = {'x': dir_x, 'y': dir_y}
    #             elif not is_in_points(x, y + 1):
    #                 direction = directions['up']
    #             else:
    #                 direction = directions['down']
    #
    #         elif direction == directions['up']:
    #             if is_in_points(x + 1, y) and not is_in_points(x, y):
    #                 current = {'x': dir_x, 'y': dir_y}
    #             elif is_in_points(x + 1, y):
    #                 direction = directions['left']
    #             else:
    #                 direction = directions['right']
    #
    #         if not is_in_cycle(current):
    #             cycle.append(current)
    #     return cycle
def create_cycle(width, height):
    cycle = []
    for y in range(height):
        if y % 2 == 0:
            for x in range(width):
                cycle.append((x, y))
        else:
            for x in range(width - 1, -1, -1):
                cycle.append((x, y))
    return cycle


import random


def get_cycle_action(head_pos, cycle, grid_width, grid_height):
    """
    Given the snake's current head position `head_pos` (x, y) and a Hamiltonian
    cycle (list of (x, y) positions covering the grid), compute the direction index
    of the move from head_pos to the next position in the cycle.

    Returns an integer: 0=up, 1=down, 2=left, 3=right.
    Raises ValueError if head_pos not in cycle or if the move is invalid.
    """
    # Ensure head_pos is a tuple of ints (x, y)
    if not isinstance(head_pos, (tuple, list)) or len(head_pos) != 2:
        raise ValueError(f"Invalid head_pos format: {head_pos}")
    x, y = head_pos
    # Check bounds of head_pos
    if not (0 <= x < grid_width and 0 <= y < grid_height):
        raise ValueError(f"Head position {head_pos} out of grid bounds")

    # Verify head_pos is in the cycle
    if head_pos not in cycle:
        raise ValueError(f"Head position {head_pos} not found in the Hamiltonian cycle")
    idx = cycle.index(head_pos)

    # Determine next position in cycle (wrap to start if at end)
    if idx == len(cycle) - 1:
        next_pos = cycle[0]
    else:
        next_pos = cycle[idx + 1]
    nx, ny = next_pos

    # Check that next_pos is within grid bounds
    if not (0 <= nx < grid_width and 0 <= ny < grid_height):
        raise ValueError(f"Next position {next_pos} out of grid bounds (wrap-around not allowed)")

    # Compute coordinate difference
    dx = nx - x
    dy = ny - y

    # Verify that move is to an adjacent cell (Manhattan distance = 1)
    if abs(dx) + abs(dy) != 1:
        raise ValueError(
            f"Invalid move from {head_pos} to {next_pos}: not adjacent"
        )

    # Map (dx, dy) to direction code
    if dx == 1 and dy == 0:
        return 3  # right
    if dx == -1 and dy == 0:
        return 2  # left
    if dx == 0 and dy == 1:
        return 1  # down
    if dx == 0 and dy == -1:
        return 0  # up

    # If none of the above, something is wrong
    raise ValueError(f"Unexpected direction delta dx={dx}, dy={dy}")

# def start_cycle(height, width):
#     BLOCK_SIZE = 20
#     grid_rows = height // BLOCK_SIZE  # 30
#     grid_cols = width // BLOCK_SIZE  # 40
#
#     print("Initializing grid...")
#     start_time = time.time()
#     prim = PrimsAlgorithm(grid_cols, grid_rows)
#     nodes = prim.create_nodes()
#     edges = prim.create_edges()
#     mst_edges = prim.create_final_edges(edges)
#
#     # This creates a cycle over the correct grid size
#     ham = HamiltonianCycle(grid_cols // 2, grid_rows // 2)
#     cycle = ham.create_cycle(nodes, mst_edges)
#     print(f"Time taken: {time.time() - start_time} seconds")
#
#     from matplotlib import pyplot as plt
#
#     # Extract x and y coordinates
#     x_coords = [point['x'] for point in cycle]
#     y_coords = [point['y'] for point in cycle]
#
#     # To close the cycle, append the first point again
#     x_coords.append(cycle[0]['x'])
#     y_coords.append(cycle[0]['y'])
#
#     # Plotting
#     plt.figure(figsize=(10, 7))
#     plt.plot(x_coords, y_coords, marker='o', linestyle='-')
#
#     # Optional: make it grid-like
#     plt.grid(True)
#     plt.title("Hamiltonian Cycle Path")
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.gca().set_aspect('equal')  # Make the grid squares proportional
#     plt.show()
#
#     cycle = [(point['x'], point['y']) for point in cycle]
#     print("Hamiltonian Cycle with MST:", cycle)
#     # print(f"First-node{cycle[0]}")
#     # print(get_cycle_action(, cycle))
#     # print(f"Second-node{cycle[1]}")
#     # print(get_cycle_action(cycle[1], cycle))
#     # print(f"Third-node{cycle[2]}")
#     # print(get_cycle_action(cycle[2], cycle))
#
#     return cycle

