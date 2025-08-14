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
                    hamiltonian_graph[j*2 + 1, i*2] = [(j*2 + 2, i*2)]
                    hamiltonian_graph[j*2 + 1, i*2 + 1] = [(j*2 + 2, i*2 + 1)]
                else:
                    hamiltonian_graph[j*2 + 1, i*2] = [(j*2 + 1, i*2 + 1)]
                if 'down' in orientation[j, i]:
                    hamiltonian_graph[j*2, i*2 + 1] = [(j*2, i*2 + 2)]
                    if (j*2 + 1, i*2 + 1) in hamiltonian_graph:
                        hamiltonian_graph[j * 2 + 1, i * 2 + 1] += [(j * 2 + 1, i * 2 + 2)]
                    else:
                        hamiltonian_graph[j*2 + 1, i*2 + 1] = [(j*2 + 1, i*2 + 2)]
                else:
                    hamiltonian_graph[j*2, i*2 + 1] = [(j*2 + 1, i*2 + 1)]
                if 'down' not in orientation[j, i-1]:
                    hamiltonian_graph[j*2, i*2] = [(j*2 + 1, i*2)]
                if 'right' not in orientation[j-1, i]:
                    if (j*2, i*2) in hamiltonian_graph:
                        hamiltonian_graph[j * 2, i * 2] += [(j * 2, i * 2 + 1)]
                    else:
                        hamiltonian_graph[j*2, i*2] = [(j*2, i*2 + 1)]

            # Finds available adjacent cells if current cell is in the bottom right corner
            elif j == grid_columns - 1 and i == grid_rows - 1:
                hamiltonian_graph[j*2, i*2 + 1] = [(j*2 + 1, i*2 + 1)]
                hamiltonian_graph[j*2 + 1, i*2] = [(j*2 + 1, i*2 + 1)]
                if 'down' not in orientation[j, i-1]:
                    hamiltonian_graph[j*2, i*2] = [(j*2 + 1, i*2)]
                elif 'right' not in orientation[j-1, i]:
                    hamiltonian_graph[j*2, i*2] = [(j*2, i*2 + 1)]

            # Finds available adjacent cells if current cell is in the top right corner
            elif j == grid_columns - 1 and i == 0:
                hamiltonian_graph[j*2, i*2] = [(j*2 + 1, i*2)]
                hamiltonian_graph[j*2 + 1, i*2] = [(j*2 + 1, i*2 + 1)]
                if 'down' in orientation[j, i]:
                    hamiltonian_graph[j*2, i*2 + 1] = [(j*2, i*2 + 2)]
                    hamiltonian_graph[j*2 + 1, i*2 + 1] = [(j*2 + 1, i*2 + 2)]
                else:
                    hamiltonian_graph[j*2, i*2 + 1] = [(j*2 + 1, i*2 + 1)]
                if 'right' not in orientation[j-1, i]:
                    hamiltonian_graph[j*2, i*2] += [(j*2, i*2 + 1)]

            # Finds available adjacent cells if current cell is in the right column
            elif j == grid_columns - 1:
                hamiltonian_graph[j*2 + 1, i*2] = [(j*2 + 1, i*2 + 1)]
                if 'down' in orientation[j, i]:
                    hamiltonian_graph[j*2, i*2 + 1] = [(j*2, i*2 + 2)]
                    hamiltonian_graph[j*2 + 1, i*2 + 1] = [(j*2 + 1, i*2 + 2)]
                else:
                    hamiltonian_graph[j*2, i*2 + 1] = [(j*2 + 1, i*2 + 1)]
                if 'down' not in orientation[j, i-1]:
                    hamiltonian_graph[j*2, i*2] = [(j*2 + 1, i*2)]
                if 'right' not in orientation[j-1, i]:
                    if (j*2, i*2) in hamiltonian_graph:
                        hamiltonian_graph[j * 2, i * 2] += [(j * 2, i * 2 + 1)]
                    else:
                        hamiltonian_graph[j*2, i*2] = [(j*2, i*2 + 1)]

            # Finds available adjacent cells if current cell is in the top left corner
            elif j == 0 and i == 0:
                hamiltonian_graph[j*2, i*2] = [(j*2 + 1, i*2)]
                hamiltonian_graph[j*2, i*2] += [(j*2, i*2 + 1)]
                if 'right' in orientation[j, i]:
                    hamiltonian_graph[j*2 + 1, i*2] = [(j*2 + 2, i*2)]
                    hamiltonian_graph[j*2 + 1, i*2 + 1] = [(j*2 + 2, i*2 + 1)]
                else:
                    hamiltonian_graph[j*2 + 1, i*2] = [(j*2 + 1, i*2 + 1)]
                if 'down' in orientation[j, i]:
                    hamiltonian_graph[j*2, i*2 + 1] = [(j*2, i*2 + 2)]
                    if (j*2 + 1, i*2 + 1) in hamiltonian_graph:
                        hamiltonian_graph[j * 2 + 1, i * 2 + 1] += [(j * 2 + 1, i * 2 + 2)]
                    else:
                        hamiltonian_graph[j*2 + 1, i*2 + 1] = [(j*2 + 1, i*2 + 2)]
                else:
                    hamiltonian_graph[j*2, i*2 + 1] = [(j*2 + 1, i*2 + 1)]

            # Finds available adjacent cells if current cell is in the bottom left corner
            elif j == 0 and i == grid_rows - 1:
                hamiltonian_graph[j*2, i*2] = [(j*2, i*2 + 1)]
                hamiltonian_graph[j*2, i*2 + 1] = [(j*2 + 1, i*2 + 1)]
                if 'right' in orientation[j, i]:
                    hamiltonian_graph[j*2 + 1, i*2] = [(j*2 + 2, i*2)]
                    hamiltonian_graph[j*2 + 1, i*2 + 1] = [(j*2 + 2, i*2 + 1)]
                else:
                    hamiltonian_graph[j*2 + 1, i*2] = [(j*2 + 1, i*2 + 1)]
                if 'down' not in orientation[j, i-1]:
                    hamiltonian_graph[j * 2, i * 2] += [(j * 2 + 1, i * 2)]

            # Finds available adjacent cells if current cell is in the left corner
            elif j == 0:
                hamiltonian_graph[j*2, i*2] = [(j*2, i*2 + 1)]
                if 'right' in orientation[j, i]:
                    hamiltonian_graph[j*2 + 1, i*2] = [(j*2 + 2, i*2)]
                    hamiltonian_graph[j*2 + 1, i*2 + 1] = [(j*2 + 2, i*2 + 1)]
                else:
                    hamiltonian_graph[j*2 + 1, i*2] = [(j*2 + 1, i*2 + 1)]
                if 'down' in orientation[j, i]:
                    hamiltonian_graph[j*2, i*2 + 1] = [(j*2, i*2 + 2)]
                    if (j*2 + 1, i*2 + 1) in hamiltonian_graph:
                        hamiltonian_graph[j*2 + 1, i*2 + 1] += [(j*2 + 1, i*2 + 2)]
                    else:
                        hamiltonian_graph[j * 2 + 1, i * 2 + 1] = [(j * 2 + 1, i * 2 + 2)]
                else:
                    hamiltonian_graph[j*2, i*2 + 1] = [(j*2 + 1, i*2 + 1)]
                if 'down' not in orientation[j, i-1]:
                    hamiltonian_graph[j*2, i*2] += [(j*2 + 1, i*2)]

            # Finds available adjacent cells if current cell is in the top row
            elif i == 0:
                hamiltonian_graph[j*2, i*2] = [(j*2 + 1, i*2)]
                if 'right' in orientation[j, i]:
                    hamiltonian_graph[j*2 + 1, i*2] = [(j*2 + 2, i*2)]
                    hamiltonian_graph[j*2 + 1, i*2 + 1] = [(j*2 + 2, i*2 + 1)]
                else:
                    hamiltonian_graph[j*2 + 1, i*2] = [(j*2 + 1, i*2 + 1)]
                if 'down' in orientation[j, i]:
                    hamiltonian_graph[j*2, i*2 + 1] = [(j*2, i*2 + 2)]
                    if (j*2 + 1, i*2 + 1) in hamiltonian_graph:
                        hamiltonian_graph[j * 2 + 1, i * 2 + 1] += [(j * 2 + 1, i * 2 + 2)]
                    else:
                        hamiltonian_graph[j*2 + 1, i*2 + 1] = [(j*2 + 1, i*2 + 2)]
                else:
                    hamiltonian_graph[j*2, i*2 + 1] = [(j*2 + 1, i*2 + 1)]
                if 'right' not in orientation[j-1, i]:
                    hamiltonian_graph[j*2, i*2] += [(j*2, i*2 + 1)]

            # Finds available adjacent cells if current cell is in the bottom row
            else:
                hamiltonian_graph[j*2, i*2 + 1] = [(j*2 + 1, i*2 + 1)]
                if 'right' in orientation[j, i]:
                    hamiltonian_graph[j*2 + 1, i*2 + 1] = [(j*2 + 2, i*2 + 1)]
                    hamiltonian_graph[j*2 + 1, i*2] = [(j*2 + 2, i*2)]
                else:
                    hamiltonian_graph[j*2 + 1, i*2] = [(j*2 + 1, i*2 + 1)]
                if 'down' not in orientation[j, i-1]:
                    hamiltonian_graph[j*2, i*2] = [(j*2 + 1, i*2)]
                if 'right' not in orientation[j-1, i]:
                    if (j*2, i*2) in hamiltonian_graph:
                        hamiltonian_graph[j*2, i*2] += [(j*2, i*2 + 1)]
                    else:
                        hamiltonian_graph[j * 2, i * 2] = [(j * 2, i * 2 + 1)]

    # Provides the coordinates of available adjacent cells to generate directions for the snake's movement
    return path_generator(hamiltonian_graph, grid_rows*grid_columns*4)

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
        elif previous_cell in graph and (previous_cell[0], previous_cell[1] + 1) in graph[previous_cell]  \
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


hamiltonian_cycle(40,40, )