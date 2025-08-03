

def generate_hamiltonian_cycle(width, height):
    cycle = []
    for y in range(height):
        if y % 2 == 0:
            for x in range(width):
                cycle.append((x, y))
        else:
            for x in reversed(range(width)):
                cycle.append((x, y))
    return cycle

def get_cycle_action(head_pos, cycle, width, height):
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

def print_cycle(width, height):
    cycle = generate_hamiltonian_cycle(width, height)
    for i, coord in enumerate(cycle):
        print(f"{i}: {coord}")

if __name__ == '__main__':
    print_cycle(40, 30)  #