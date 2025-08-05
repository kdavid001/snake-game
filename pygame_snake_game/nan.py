class HamiltonianCycle:
    def __init__(self, columns, rows):
        self.columns = columns
        self.rows = rows

    def create_cycle(self, nodes, final_edges):
        points = []

        for edge in final_edges:
            start = dict(nodes[edge['startNode']])
            end = dict(nodes[edge['endNode']])
            mid_x = (start['x'] + end['x']) // 2
            mid_y = (start['y'] + end['y']) // 2
            mid = {'x': mid_x, 'y': mid_y}
            points.extend([start, mid, end])

        def is_in_points(x, y):
            return any(pos['x'] == x and pos['y'] == y for pos in points)

        def is_in_cycle(current, cycle):
            return any(pos['x'] == current['x'] and pos['y'] == current['y'] for pos in cycle)

        cycle = [{'x': 0, 'y': 0}]
        current = cycle[0]

        directions = {
            'up': {'x': 0, 'y': -1},
            'down': {'x': 0, 'y': 1},
            'left': {'x': -1, 'y': 0},
            'right': {'x': 1, 'y': 0}
        }

        direction = directions['right']

        while len(cycle) < self.columns * self.rows:
            x, y = current['x'], current['y']
            dir_x = direction['x'] + x
            dir_y = direction['y'] + y

            if direction == directions['right']:
                if is_in_points(x + 1, y + 1) and not is_in_points(x + 1, y):
                    current = {'x': dir_x, 'y': dir_y}
                elif is_in_points(x, y + 1) and not is_in_points(x + 1, y + 1):
                    direction = directions['down']
                else:
                    direction = directions['up']

            elif direction == directions['down']:
                if is_in_points(x, y + 1) and not is_in_points(x + 1, y + 1):
                    current = {'x': dir_x, 'y': dir_y}
                elif is_in_points(x, y + 1) and is_in_points(x + 1, y + 1):
                    direction = directions['right']
                else:
                    direction = directions['left']

            elif direction == directions['left']:
                if is_in_points(x, y) and not is_in_points(x, y + 1):
                    current = {'x': dir_x, 'y': dir_y}
                elif not is_in_points(x, y + 1):
                    direction = directions['up']
                else:
                    direction = directions['down']

            elif direction == directions['up']:
                if is_in_points(x + 1, y) and not is_in_points(x, y):
                    current = {'x': dir_x, 'y': dir_y}
                elif is_in_points(x + 1, y):
                    direction = directions['left']
                else:
                    direction = directions['right']

            if not is_in_cycle(current, cycle):
                cycle.append(current)

        return cycle


from matplotlib import pyplot as plt

cycle = HamiltonianCycle(6, 6)


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

if __name__ == '__main__':
    if __name__ == '__main__':
        # Grid size
        width, height = 6, 6

        # Generate dummy nodes and final edges for testing
        nodes = []
        node_map = {}
        index = 0
        for y in range(height):
            for x in range(width):
                node = {'x': x, 'y': y}
                nodes.append(node)
                node_map[(x, y)] = index
                index += 1

        final_edges = []
        for (x, y), idx in node_map.items():
            for dx, dy in [(1, 0), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in node_map:
                    final_edges.append({'startNode': idx, 'endNode': node_map[(nx, ny)]})

        # Generate the cycle
        raw_cycle = cycle.create_cycle(nodes, final_edges)

        # Print the cycle as (x, y) tuples
        tuple_cycle = [(pos['x'], pos['y']) for pos in raw_cycle]
        print("Generated Hamiltonian cycle:")
        print(tuple_cycle)

        # Optional: Visualize it
        draw_cycle(tuple_cycle, width, height)