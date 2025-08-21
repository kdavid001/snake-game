"""
Not in use!!!!
"""
import matplotlib.pyplot as plt


def get_neighbors(cell, width, height):
    """Get all valid neighboring cells"""
    x, y = cell
    neighbors = []
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            neighbors.append((nx, ny))
    return neighbors


def create_base_cycle(width, height):
    """Create a simple snake pattern cycle as base"""
    cycle = []
    for y in range(height):
        if y % 2 == 0:
            cycle.extend((x, y) for x in range(width))
        else:
            cycle.extend((x, y) for x in reversed(range(width)))

    # Add return path
    if height % 2 == 1:
        cycle.extend((width - 1, y) for y in reversed(range(height - 1)))
    else:
        cycle.extend((0, y) for y in reversed(range(height - 1)))

        cycle.append(cycle[0])
    return cycle


import random


def is_valid_cycle(cycle):
    for i in range(len(cycle)):
        a = cycle[i]
        b = cycle[(i + 1) % len(cycle)]
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return False
    return True


def randomize_cycle(cycle, width, height, iterations=1000):
    for _ in range(iterations):
        i = random.randint(0, len(cycle) - 3)
        max_j = len(cycle) - 1
        if i + 2 > max_j:
            continue
        j = random.randint(i + 2, min(i + 10, max_j))
        if j == len(cycle) - 1 and i == 0:
            continue
        new_cycle = cycle[:i + 1] + cycle[i + 1:j + 1][::-1] + cycle[j + 1:]
        if is_valid_cycle(new_cycle):
            cycle = new_cycle
    return cycle


def generate_random_hamiltonian(width, height):
    cycle = create_base_cycle(width, height)
    randomized_cycle = randomize_cycle(cycle, width, height)
    if is_valid_cycle(randomized_cycle):
        return randomized_cycle
    else:
        print("Warning: Final randomized cycle is invalid. Returning base cycle.")
        return cycle


# New function to get actions from cycle
def get_cycle_action(head_pos, cycle):
    try:
        idx = cycle.index(head_pos)
        next_pos = cycle[(idx + 1) % len(cycle)]
    except ValueError:
        print(f"[ERROR] Head position {head_pos} not in Hamiltonian cycle!")
        return None

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
        return None


def draw_cycle(cycle, width, height):
    """Visualize the cycle"""
    fig, ax = plt.subplots(figsize=(10, 10))

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


# Example usage
if __name__ == "__main__":
    width, height = 20, 20
    random_cycle = generate_random_hamiltonian(width, height)
    print(f"Generated {len(random_cycle) - 1} moves (including return to start)")
    draw_cycle(random_cycle, width, height)
