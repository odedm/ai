# kruskal.py
# ---------------
# An implementation of Kruskal's algorithm.
# Used to find a minimal spanning tree.
# Based on an algorithm found online.

parent = dict()
rank = dict()

# Helper methods for Kruskal union-find

def make_set(vertice):
    parent[vertice] = vertice
    rank[vertice] = 0

def find(vertice):
    if parent[vertice] != vertice:
        parent[vertice] = find(parent[vertice])
    return parent[vertice]

def union(vertice1, vertice2):
    root1 = find(vertice1)
    root2 = find(vertice2)
    if root1 != root2:
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
            parent[root1] = root2
            if rank[root1] == rank[root2]: rank[root2] += 1

def kruskal(graph):
    """ Returns the MST of the given graph """
    for vertice in graph.vertices:
        make_set(vertice)

    minimum_spanning_tree = set()
    edges = list(graph.edges)
    edges.sort()
    for edge in edges:
        weight, vertice1, vertice2 = edge
        if find(vertice1) != find(vertice2):
            union(vertice1, vertice2)
            minimum_spanning_tree.add(edge)
    return minimum_spanning_tree

class Graph(object):
    """ Represents an abstract graph.
    Example graph:
    graph = {
        vertices: ['A', 'B', 'C', 'D', 'E', 'F'],
        edges: set([
            (1, 'A', 'B'),
            (5, 'A', 'C'),
            ])
        }
    """
    
    def __init__(self, vertices):
        self.vertices = vertices
        self.edges = set()
        
    def add_edge(self, v1, v2, w):
        if v1 != v2:
            self.edges.add((w, v1, v2))
