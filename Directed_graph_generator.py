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
        # x = rng.choice(seq)
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
    # G = nx.empty_graph(m)
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
        # print(tuple(zip([source] * m, targets)))
        t = tuple(zip([source] * m, targets))
        t2 = np.array(t)
        nt2 = len(t2)
        both = []
        print("t2  ", t2)
        for i in range(0, nt2):
            x = random.random()
            if x < 1 / 3:  # with probability 1/3 the direction of the link changes
                temp = t2[i][0]
                t2[i][0] = t2[i][1]
                t2[i][1] = temp
            if x > 2 / 3:  # with probability 1/3 both link are added to the network
                both.append([t2[i][1], t2[i][0]])
        print("t2after   ", t2)
        G.add_edges_from(t2)
        G.add_edges_from(both)
        # G.add_edges_from(zip([source] * m, targets))
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
    # for value in edges:
    #    print(value)
    # print("ed: ",edges)
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
    # n_name, nodes = n
    n_name = n
    nodes = range(0, n)
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    # G.add_nodes_from(nodes)
    # If no positions are provided, choose uniformly random vectors in
    # Euclidean space of the specified dimension.
    if pos is None:
        # pos = {v: [seed.random() for i in range(dim)] for v in nodes}
        pos = {v: [random.random() for i in range(dim)] for v in nodes}

    nx.set_node_attributes(G, pos, "pos")

    if _is_scipy_available:
        edges = _fast_edges(G, radius, p)
    else:
        edges = _slow_edges(G, radius, p)
    links = []
    for value in edges:
        e = list(value)
        x = random.random()
        if x < 1 / 3:  # with probability 1/3 the direction of the link changes
            e.reverse()
        if x > 2 / 3:  # with probability 1/3 both link are added to the network
            rev = [e[1], e[0]]
            links.append(rev)
        links.append(e)
    G.add_edges_from(links)
    return G


def barabasi_albert_directed_graph_with_probabilities(n, m, p1=1/3, p2=1/3, seed=None):
    """ BARABASI-ALBERT MODEL WITH USER-RANDOM DIRECTION ASSIGNMENT FOR EACH LINK
        p1 = probability that the link is directed to the new node -> HUBS (default 1/3)
        p2 = probability that the link is directed to an already existing node -> AUTHORITY (default 1/3)
        1-p1-p2 = probability that the new node is connected with both links (default 1/3)
    """
    if m < 1 or m >= n:
        raise nx.NetworkXError(
            f"Barabási–Albert network must have m >= 1 and m < n, m = {m}, n = {n}"
        )

    # Add m initial nodes (m0 in barabasi-speak)
    # G = nx.empty_graph(m)
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
        # print(tuple(zip([source] * m, targets)))
        t = tuple(zip([source] * m, targets))
        t2 = np.array(t)
        nt2 = len(t2)
        both = []
        for i in range(0, nt2):
            x = random.random()
            if x < p1:  # with probability p1 the direction of the link changes
                temp = t2[i][0]
                t2[i][0] = t2[i][1]
                t2[i][1] = temp
            if x > p1 + p2:  # with probability 1-p1-p2 both link are added to the network
                both.append([t2[i][1], t2[i][0]])
        G.add_edges_from(t2)
        G.add_edges_from(both)
        # G.add_edges_from(zip([source] * m, targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = _random_subset(repeated_nodes, m, seed)
        source += 1
    return G
