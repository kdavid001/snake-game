import time
import random


class HamiltonianCycle:
    def __init__(self, columns, rows):
        self.columns = columns
        self.rows = rows

    def create_cycle(self):
        cycle = []
        for y in range(self.rows):
            row = range(self.columns) if y % 2 == 0 else reversed(range(self.columns))
            for x in row:
                cycle.append((x, y))
        return cycle


def get_cycle_action(head_pos, cycle):
    try:
        idx = cycle.index(head_pos)
        next_pos = cycle[(idx + 1) % len(cycle)]
    except ValueError:
        print(f"[ERROR] Head position {head_pos} not in Hamiltonian cycle!")
        return random.randint(0, 3)

    dx = next_pos[0] - head_pos[0]
    dy = next_pos[1] - head_pos[1]

    if dx == 1:
        return 0  # RIGHT
    elif dx == -1:
        return 1  # LEFT
    elif dy == 1:
        return 2  # DOWN
    elif dy == -1:
        return 3  # UP
    else:
        print(f"[ERROR] Invalid movement from {head_pos} to {next_pos}")
        return random.randint(0, 3, )  # if no valid path, choose randomly
        # Initialize grid just incase the snake body drifts from the grid


# Initialize grid
print("Initializing grid...")
start_time = time.time()

ham = HamiltonianCycle(40, 30)
cycle = ham.create_cycle()

print("Hamiltonian Cycle:", cycle)
print(f"Time taken: {time.time() - start_time} seconds")

from matplotlib import pyplot as plt

# Extract x and y coordinates
x_coords = [x for x, y in cycle]
y_coords = [y for x, y in cycle]

# To close the cycle, append the first point again
x_coords.append(cycle[0][0])
y_coords.append(cycle[0][1])

# Plotting
plt.figure(figsize=(10, 7))
plt.plot(x_coords, y_coords, marker='o', linestyle='-')

# Optional: make it grid-like
plt.grid(True)
plt.title("Hamiltonian Cycle Path")
plt.xlabel("X")
plt.ylabel("Y")
plt.gca().set_aspect('equal')  # Make the grid squares proportional
plt.show()
