import numpy as np
import random
import itertools
import networkx as nx
from scipy.spatial import KDTree


try:
    from scipy.spatial import cKDTree as KDTree
except ImportError:
    _is_scipy_available = False
else:
    _is_scipy_available = True



def _random_subset(seq, m, rng):
    """ Return m unique elements from seq.
    """
    targets = set()
    while len(targets) < m:
        x = random.choice(seq)
        #x = rng.choice(seq)
        targets.add(x)
    return targets


def barabasi_albert_directed_graph(n, m, seed=None):
    """ BARABASI-ALBERT MODEL WITH RANDOM DIRECTION ASSIGNMENT FOR EACH LINK
    """
    if m < 1 or m >= n:
        raise nx.NetworkXError(
            f"Barabási–Albert network must have m >= 1 and m < n, m = {m}, n = {n}"
        )

    # Add m initial nodes (m0 in barabasi-speak)
    #G = nx.empty_graph(m)
    G = nx.DiGraph()
    G.add_nodes_from(range(0, m))
    # Target nodes for new edges
    targets = list(range(m))
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = []
    # Start adding the other n-m nodes. The first node is m.
    source = m
    while source < n:
        # Add edges to m nodes from the source.
        #print(tuple(zip([source] * m, targets)))
        t = tuple(zip([source] * m, targets))
        t2 = np.array(t)
        #print("1 :",t2)
        for i in range(0,len(t2)):
            x = random.random()
            if x < 0.5:
                temp = t2[i][0]
                t2[i][0] = t2[i][1]
                t2[i][1] = temp
        #print("2: ",t2)
        G.add_edges_from(t2)
        #G.add_edges_from(zip([source] * m, targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = _random_subset(repeated_nodes, m, seed)
        source += 1
    return G



def _fast_edges(G, radius, p):
    """Returns edge list of node pairs within `radius` of each other
       using scipy KDTree and Minkowski distance metric `p`

    Requires scipy to be installed.
    """
    pos = nx.get_node_attributes(G, "pos")
    nodes, coords = list(zip(*pos.items()))
    kdtree = KDTree(coords)  # Cannot provide generator.
    edge_indexes = kdtree.query_pairs(radius, p)
    edges = ((nodes[u], nodes[v]) for u, v in edge_indexes)
    #for value in edges:
    #    print(value)
    #print("ed: ",edges)
    return edges


def _slow_edges(G, radius, p):
    """Returns edge list of node pairs within `radius` of each other
       using Minkowski distance metric `p`

    Works without scipy, but in `O(n^2)` time.
    """
    # TODO This can be parallelized.
    edges = []
    for (u, pu), (v, pv) in itertools.combinations(G.nodes(data="pos"), 2):
        if sum(abs(a - b) ** p for a, b in zip(pu, pv)) <= radius ** p:
            edges.append((u, v))
    return edges


def random_geometric_directed_graph(n, radius, dim=2, pos=None, p=2, seed=None):
    """ GEOMETRIC MODEL WITH RANDOM DIRECTION ASSIGNMENT FOR EACH LINK
        """
    #n_name, nodes = n
    n_name = n
    nodes = range(0, n)
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    #G.add_nodes_from(nodes)
    # If no positions are provided, choose uniformly random vectors in
    # Euclidean space of the specified dimension.
    if pos is None:
        #pos = {v: [seed.random() for i in range(dim)] for v in nodes}
        pos = {v: [random.random() for i in range(dim)] for v in nodes}

    nx.set_node_attributes(G, pos, "pos")

    if _is_scipy_available:
        edges = _fast_edges(G, radius, p)
    else:
        edges = _slow_edges(G, radius, p)
    ed = []
    for value in edges:
        #print("v ",value)
        e = list(value)
        x = random.random()
        if x < 0.5:
            e.reverse()
        #print("e ",e)
        ed.append(e)
    G.add_edges_from(ed)

    return G
