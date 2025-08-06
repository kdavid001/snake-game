import random

class PrimsAlgorithm:
    def __init__(self, columns, rows):
        self.columns = columns // 2
        self.rows = rows // 2

    # Create 2x2 block-centered nodes
    def create_nodes(self):
        nodes = []
        for y in range(self.rows):
            for x in range(self.columns):
                nodes.append({'x': x * 2 + 1, 'y': y * 2 + 1})
        return nodes

    # Create weighted edges between adjacent nodes
    def create_edges(self):
        def rand_weight():
            return random.randint(1, 3)

        edges = []

        # Horizontal edges
        for y in range(self.rows):
            for x in range(self.columns - 1):
                start_node = y * self.columns + x
                end_node = start_node + 1
                edges.append({'startNode': start_node, 'endNode': end_node, 'weight': rand_weight()})

        # Vertical edges
        for x in range(self.columns):
            for y in range(self.rows - 1):
                start_node = y * self.columns + x
                end_node = start_node + self.columns
                edges.append({'startNode': start_node, 'endNode': end_node, 'weight': rand_weight()})

        return edges

    # Apply Prim's algorithm to choose minimal edges
    def create_final_edges(self, edges):
        unvisited = list(range(self.columns * self.rows))
        visited = []
        current = 0
        final_edges = []

        while unvisited:
            unvisited = [node for node in unvisited if node != current]
            visited.append(current)

            my_edges = []
            for edge in edges:
                visited_s = edge['startNode'] in visited
                visited_e = edge['endNode'] in visited

                if visited_s and visited_e:
                    continue
                if visited_s or visited_e:
                    my_edges.append(edge)

            min_edge = {'weight': float('inf')}
            for edge in my_edges:
                if edge['weight'] < min_edge['weight']:
                    min_edge = edge

            if not unvisited:
                break

            final_edges.append(min_edge)

            if min_edge['weight'] == float('inf'):
                current = unvisited[0]
            elif min_edge['endNode'] in visited:
                current = min_edge['startNode']
            else:
                current = min_edge['endNode']

        return final_edges


if __name__ == "__main__":
    prim = PrimsAlgorithm(80, 60)
    nodes = prim.create_nodes()
    edges = prim.create_edges()
    mst = prim.create_final_edges(edges)

    print("Nodes:", nodes)
    print("Final MST edges:", mst)
