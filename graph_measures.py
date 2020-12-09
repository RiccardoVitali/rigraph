import numpy as np
import scipy
from scipy.linalg import expm
import pykov
import random
import networkx as nx


def check_positivity_element_wise(self):
    res = self <= 0
    for x in res:
        for j in x:
            if j:
                return False
    return True


def check_irreducibility(self):
    n = self.shape[0]
    return check_positivity_element_wise(np.linalg.matrix_power(np.identity(n) + self, n - 1))


def to_stochastic_matrix(self):  # prende in argomento la matrice W e ritorna la matrice stocastica associata
    rs = []
    for i in self:
        y = []
        rs.append(y)
        for j in i:
            if np.sum(i) > 0:
                x = j / np.sum(i)
                y.append(x)
            else:
                y.append(0)
    return rs


def steady_state_distribution(self):
    return to_mc(to_stochastic_matrix(self)).steady()


def MFPT(self):  # prende in argomento la matrice W e ritorna la matrice dei mean first passage time
    n = self.shape[0]
    rs = []
    for j in range(0, n):
        s = to_stochastic_matrix(self)
        s1 = np.delete(s, j, 0)
        s2 = np.delete(s1, j, 1)
        rs = np.append(rs, np.dot(scipy.linalg.inv(np.identity(n - 1) - s2), np.ones(n - 1)))
        rs = np.insert(rs, j * (n + 1), 0)
    return (np.reshape(rs, [n, n])).transpose()


def MFPT_from_stoc(self):  # prende in argomento una matrice stocastica e ritorna la matrice dei mean first passage time
    n = len(self)
    rs = []
    for j in range(0, n):
        s1 = np.delete(self, j, 0)
        s2 = np.delete(s1, j, 1)
        rs = np.append(rs, np.dot(scipy.linalg.inv(np.identity(n - 1) - s2), np.ones(n - 1)))
        rs = np.insert(rs, j * (n + 1), 0)
    return (np.reshape(rs, [n, n])).transpose()


def rwc_centrality_per_node(self, x):  # prende in argomento un nodo e la matrice W e ritorna un numero
    res = 0
    n = self.shape[0]
    for j in range(0, n):
        if j != x:
            res += MFPT(self)[j][x]
    return (n - 1) / res


def rwc_centrality(self):  # prende in argomento la matrice W e da la random walk closeness centrality di ogni nodo
    res = []
    for i in range(0, self.shape[0]):
        res.append(rwc_centrality_per_node(self, i))
    return res


def to_mc(self):  # prende in argomento una matrice stocastica e la transforma in una catena di markov
    n = len(self)
    mcs = {}
    for i in range(0, n):
        for j in range(0, n):
            mcs[(i, j)] = self[i][j]
    return pykov.Chain(mcs)


def kemeny_constant(self):  # prende in argomento la matrice W e ritorna un numero
    return to_mc(to_stochastic_matrix(self)).kemeny_constant()


def communicability(self):  # prende in argomento la matrice W e ritorna la matrice della communicability
    return expm(self)


def estrada_index(self):  # prende in argomento la matrice W e ritorna un numero
    n = len(self)
    return np.trace(expm(self))


def estrada_01(self):
    n = len(self)
    x = np.diag(expm(self))
    x_min = np.min(x)
    x_max = np.max(x)
    s = 0
    for i in range(0, n):
        s += (x[i] - x_min) / (x_max - x_min)
    return s / n


def communicability_01(self):
    n = len(self)
    comm = expm(self)
    c_max = comm.max()
    c_min = comm.min()
    s = 0
    for i in range(0, n):
        for j in range(0, n):
            s += (comm[i][j] - c_min) / (c_max - c_min)
    return s / (n * n)


def out_strengths(self):  # prende in argomente la matrice W e ritorna il vettore delle s^out
    lista_s_out = []
    n = len(self)
    for i in range(0, n):
        x = 0
        for j in range(0, n):
            x += self[i][j]
        lista_s_out.append(x)
    return lista_s_out


# prende in argomento la matrice W e ritorna un numero
def total_comm_for_unweighted(self):
    n = len(self)
    return np.dot(np.ones(n), (np.dot(communicability(self), np.ones(n).transpose())))


def total_comm_for_weighted(
        self):  # prende in argomento la matrice W e ritorna un numero (!! normalizzata per D^-0.5 !!)
    n = len(self)
    s_out = out_strengths(self)
    diag = np.diag(s_out)
    D = np.power(np.linalg.inv(diag), 0.5)
    W_tilde = np.matmul(D, np.matmul(self, D))
    return np.dot(np.ones(n), (np.dot(communicability(W_tilde), np.ones(n).transpose()))) / n


def total_hub_communicability(self):  # prende in argomento la matrice W e ritorna un numero
    HC = np.cosh(np.sqrt(np.dot(self, self.transpose())))
    return np.dot(np.ones(HC.shape[0]), (np.dot(HC, np.ones(HC.shape[0]).transpose())))


def total_authority_communicability(self):  # prende in argomento la matrice W e ritorna un numero
    AC = np.cosh(np.sqrt(np.dot(self.transpose(), self)))
    return np.dot(np.ones(AC.shape[0]), (np.dot(AC, np.ones(AC.shape[0]).transpose())))


def piccardi_distance_sloooow(self,
                              T):  # prende in argomente la matrice W e la lunghezza della random walk T e ritorna un numero
    P = to_stochastic_matrix(self)
    similarities = []
    for i in range(0, len(self)):
        for j in range(i + 1, len(self)):
            s = 0
            for t in range(1, T + 1):
                x = np.linalg.matrix_power(P, t)[i, j]
                y = np.linalg.matrix_power(P, t)[j, i]
                z = x + y
                s += z
            similarities.append(s)
    if np.max(similarities) == np.min(similarities):
        return similarities[0]
    distances = []
    for i in range(0, len(similarities)):
        distances.append(1 - (similarities[i] - np.min(similarities)) / (np.max(similarities) - np.min(similarities)))

    return 1 - np.sum(distances) / len(distances)


def piccardi_distance(self,
                      T):  # prende in argomente la matrice W e la lunghezza della random walk T e ritorna un numero
    n = len(self)
    p = to_stochastic_matrix(self)
    pt_ij = p
    pt_ji = np.transpose(p)
    similarities = []
    power = p
    for t in range(2, T + 1):
        power = np.matmul(power, p)
        pt_ij += power
        pt_ji += np.transpose(power)
    p = pt_ij + pt_ji
    for i in range(0, n):
        for j in range(i + 1, n):
            similarities.append(p[i, j])
    ns = len(similarities)
    mins = np.min(similarities)
    maxs = np.max(similarities)
    if maxs == mins:
        if similarities[0] > 1:
            return 1
        else:
            return similarities[0]
    sumd = 0
    for i in range(0, ns):
        sumd += 1 - (similarities[i] - mins) / (maxs - mins)
    return 1 - sumd / ns


# iteraz è il numero di sample che prendo per fare la media (per n grande sarebbe troppo dispendioso fare iteraz = n)
def pons_latapy_distance(self, T, iteraz=30):
    n = len(self)
    if n < iteraz:
        return "iteraz cannot be greater than n"
    p = to_stochastic_matrix(self)
    k = out_strengths(self)
    z = 0
    while z < n:
        if k[z] == 0:
            return "It is not possible to compute pons_latapy distance since the graph is not connected"
        z += 1
    power = np.linalg.matrix_power(p, T)
    s = 0
    sumd = 0
    l = []
    while len(l) < iteraz:
        numero = random.randint(0, n - 1)
        if numero not in l:
            l.append(numero)
    for i in l:
        for j in l:
            if j != i:
                for h in range(0, n):
                    s += np.power(power[i, h] - power[h, j], 2) / k[h]
                sumd += np.sqrt(s)
    return sumd / (iteraz*(iteraz-1))


def comm_invariant(self):
    n = len(self)
    g = expm(self)
    s1 = np.dot(np.diag(g), np.transpose(np.ones(n)))
    s2 = np.dot(np.ones(n), np.transpose(np.diag(g)))
    x = 0.5 * np.sqrt(s1 + s2 - 2 * g)
    return np.dot(np.transpose(np.ones(n)), np.dot(x, np.ones(n)))


def comm_invariant_pairs(self):
    n = len(self)
    g = expm(self)
    s1 = np.dot(np.diag(g), np.transpose(np.ones(n)))
    s2 = np.dot(np.ones(n), np.transpose(np.diag(g)))
    x = 0.5 * np.sqrt(s1 + s2 - 2 * g)
    pairs = []
    for i in range(0, n):
        for j in range(0, n):
            if i != j:
                pairs.append(x[i][j])
    return pairs


def comm_invariant_pairs_triangular(self):
    n = len(self)
    g = expm(self)
    s1 = np.dot(np.diag(g), np.transpose(np.ones(n)))
    s2 = np.dot(np.ones(n), np.transpose(np.diag(g)))
    x = 0.5 * np.sqrt(s1 + s2 - 2 * g)
    pairs = []
    for i in range(0, n):
        for j in range(i+1, n):
            pairs.append(x[i][j])
    return pairs


def comm_invariant_01(self):
    n = len(self)
    g = expm(self)
    s1 = np.dot(np.diag(g), np.transpose(np.ones(n)))
    s2 = np.dot(np.ones(n), np.transpose(np.diag(g)))
    x = np.sqrt(s1 + s2 - 2 * g)
    x_min = x.min()
    x_max = x.max()
    x = (x - x_min) / (x_max - x_min)
    return np.dot(np.transpose(np.ones(n)), np.dot(x, np.ones(n))) / (n * n)


def directed_efficiency(self):  # prende un grafo e ritorna la sua efficiency
    sp = nx.shortest_path(self)
    n = nx.number_of_nodes(self)
    s = 0
    for i in range(0, len(sp)):
        for x in sp[i].values():
            # print(x)
            if len(x) > 1:
                s += 1 / (len(x) - 1)
    return s / (n * (n - 1))


def shortest_path_pairs_triangular(self):#prende un grafo e ritorna il vettore delle distanze di tutte le coppie di nodi: n(n-1)/2
    d_ij = []
    n = nx.number_of_nodes(self)
    sp = dict(nx.all_pairs_shortest_path_length(self))
    for i in range(0, n):
        for j in range(i + 1, n):
            d_ij.append(sp[i][j])
    return d_ij


def efficiency_pairs_triangular(self):#prende un grafo e ritorna il vettore delle efficiency di tutte le coppie di nodi: n(n-1)/2
    e_ij = []
    n = nx.number_of_nodes(self)
    sp = dict(nx.all_pairs_shortest_path_length(self))
    for i in range(0, n):
        for j in range(i + 1, n):
            e_ij.append(1/sp[i][j])
    return e_ij


def shortest_path_pairs(self):#prende un grafo e ritorna il vettore delle distanze di tutte le coppie di nodi: n(n-1)
    d_ij = []
    n = nx.number_of_nodes(self)
    sp = dict(nx.all_pairs_shortest_path_length(self))
    for i in range(0, n):
        for j in range(0, n):
            if i != j:
                d_ij.append(sp[i][j])
    return d_ij


def efficiency_pairs(self):#prende un grafo e ritorna il vettore delle efficiency di tutte le coppie di nodi: n(n-1)
    e_ij = []
    n = nx.number_of_nodes(self)
    sp = dict(nx.all_pairs_shortest_path_length(self))
    for i in range(0, n):
        for j in range(0, n):
            if i != j:
                e_ij.append(1/sp[i][j])
    return e_ij


def MFPT_pairs(self): # prende la matrice W e ritorna il vettore dei MFPT di tutte le coppie di nodi: n(n-1)
    n = len(self)
    m = MFPT(self)
    pairs = []
    for i in range(0, n):
        for j in range(0, n):
            if i != j:
                pairs.append(m[i][j])
    return pairs


def communicability_pairs_triangular(self):
    c = communicability(self)
    pairs = []
    n = len(self)
    for i in range(0, n):
        for j in range(i+1, n):
            pairs.append(c[i][j])
    return pairs


def communicability_pairs(self):
    c = communicability(self)
    pairs = []
    n = len(self)
    for i in range(0, n):
        for j in range(0, n):
            if i != j:
                pairs.append(c[i][j])
    return pairs


def log_communicability_pairs(self):
    return np.log(communicability_pairs(self))

# prende in argomente la matrice W e la lunghezza della random walk T e ritorna il vettore n(n-1)/2
def piccardi_distance_pairs_triangular(self,T):
    n = len(self)
    p = to_stochastic_matrix(self)
    pt_ij = p
    pt_ji = np.transpose(p)
    similarities = []
    power = p
    for t in range(2, T + 1):
        power = np.matmul(power, p)
        pt_ij += power
        pt_ji += np.transpose(power)
    p = pt_ij + pt_ji
    for i in range(0, n):
        for j in range(i + 1, n):
            similarities.append(p[i, j])
    mis = np.min(similarities)
    mas = np.max(similarities)
    if mas == mis:
        if similarities[0] > 1:
            return 1
        else:
            return similarities[0]
    pairs = []
    sl = len(similarities)
    for i in range(0, sl):
        pairs.append(similarities[i])
    return pairs


# prende in argomente la matrice W e la lunghezza della random walk T e ritorna il vettore n(n-1)
def piccardi_distance_pairs(self, T):
    n = len(self)
    p = to_stochastic_matrix(self)
    pt_ij = p
    pt_ji = np.transpose(p)
    similarities = []
    power = p
    for t in range(2, T + 1):
        power = np.matmul(power, p)
        pt_ij += power
        pt_ji += np.transpose(power)
    p = pt_ij + pt_ji
    for i in range(0, n):
        for j in range(0, n):
            if i != j:
                similarities.append(p[i, j])
    mis = np.min(similarities)
    mas = np.max(similarities)
    if mas == mis:
        if similarities[0] > 1:
            return 1
        else:
            return similarities[0]
    pairs = []
    sl = len(similarities)
    for i in range(0, sl):
        pairs.append(similarities[i])
    return pairs


def pons_latapy_distance_pairs(self, T):
    n = len(self)
    p = to_stochastic_matrix(self)
    k = out_strengths(self)
    z = 0
    while z < n:
        if k[z] == 0:
            return "It is not possible to compute pons_latapy distance since the graph is not connected"
        z += 1
    power = np.linalg.matrix_power(p, T)
    s = 0
    pairs = []
    for i in range(0, n):
        for j in range(0, n):
            if j != i:
                for h in range(0, n):
                    s += np.power(power[i, h] - power[h, j], 2) / k[h]
                pairs.append(np.sqrt(s))
    return pairs


