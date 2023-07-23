from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import SpectralEmbedding
from pynndescent import NNDescent
from typing import Tuple, List, Union
from scipy.optimize import curve_fit
from umap.spectral import spectral_layout

import numpy as np
import scipy
import numba as nb


def spectral_embedding(
    high_dimensional_space: np.ndarray,
    topological_representation: np.ndarray,
    dimensions: int
) -> np.ndarray:
    initialisation = spectral_layout(
        data=high_dimensional_space,
        graph=scipy.sparse.csr_matrix(topological_representation),
        dim=dimensions,
        random_state=np.random.RandomState(42),
        metric='euclidean',
        metric_kwds={})
    expansion = 10.0 / np.abs(initialisation).max()
    embedding = (initialisation * expansion).astype(np.float32)
    embedding = (
        10.0 * (embedding - np.min(embedding, 0)) / (np.max(embedding, 0) - np.min(embedding, 0))
    ).astype(np.float32, order="C")
    return embedding


@nb.njit(parallel=True)
def fast_knn_indices(pairwise_distances: np.ndarray, n_neighbors: int):
    knn_indices = np.empty((pairwise_distances.shape[0], n_neighbors), dtype=np.int32)
    for row in nb.prange(pairwise_distances.shape[0]):
        v = pairwise_distances[row].argsort(kind="quicksort")
        v = v[:n_neighbors]
        knn_indices[row] = v
    return knn_indices


def approx_nearest_neighbors(pairwise_distances: np.ndarray, neighbors_count: int) -> Tuple[np.ndarray, np.ndarray]:
    knn_indices = fast_knn_indices(pairwise_distances, neighbors_count)
    knn_dists = pairwise_distances[np.arange(pairwise_distances.shape[0])[:, None], knn_indices].copy()
    return knn_indices, knn_dists


def psi_fit(min_dist: float):
    xv = np.linspace(0, 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist))
    functional = lambda x, a, b: 1.0 / (1.0 + a * x ** (2 * b))
    params, covar = curve_fit(functional, xv, yv)
    return params[0], params[1]


@nb.njit(fastmath=True, parallel=True)
def exponential_differences(distances: np.ndarray, p: np.float32, sigma: float) -> np.ndarray:
    difference = -1 * distances + p
    return np.exp(difference / sigma)


@nb.njit(fastmath=True, parallel=True)
def k(distances: np.ndarray, p: np.float32, sigma: float) -> np.float64:
    probabilities = exponential_differences(distances, p, sigma)
    summa = np.sum(probabilities)
    return np.power(2, np.minimum(summa, 1000))


@nb.njit
def sigma_binary_search(distances: np.ndarray, p: np.float32, fixed_k: int) -> float:
    sigma_lower = 2
    sigma_upper = 100
    for iteration in range(20):
        approx_sigma = (sigma_lower + sigma_upper) / 2
        approx_sigma_k = k(distances, p, approx_sigma)
        if np.abs(fixed_k - approx_sigma_k) <= 1e-5:
            break
        if approx_sigma_k < fixed_k:
            sigma_lower = approx_sigma
        else:
            sigma_upper = approx_sigma
    return approx_sigma


@nb.njit
def local_fuzzy_simplicial_set(
    n: int,
    neighbors_count: int,
    knn_indices: np.ndarray,
    knn_dists: np.ndarray) -> np.ndarray:
    fuzzy_sets = np.zeros((n, n))
    for index in range(n):
        neighbor_indexes = knn_indices[index][1:neighbors_count]
        neighbor_distances = knn_dists[index][1:neighbors_count]
        p_i = neighbor_distances[0]
        sigma = sigma_binary_search(neighbor_distances, p_i, neighbors_count)
        fuzzy_distances = exponential_differences(neighbor_distances, p_i, sigma)
        fuzzy_sets[index][neighbor_indexes] = fuzzy_distances
    return fuzzy_sets


@nb.njit
def local_fuzzy_simplicial_type2_set(
    n: int,
    knn_indices: np.ndarray,
    knn_dists: np.ndarray,
    neighbor_counts: np.ndarray) -> np.ndarray:
    fuzzy = np.zeros((n, n))
    denominator = len(neighbor_counts)
    for k in range(denominator):
        neighbors_count = neighbor_counts[k]
        fuzzy_set = local_fuzzy_simplicial_set(n, neighbors_count, knn_indices, knn_dists)
        fuzzy = fuzzy + fuzzy_set
    return fuzzy / denominator


@nb.njit(fastmath=True, parallel=True)
def probabilistic_t_conorm(fuzzy_sets: np.ndarray):
    transpose = np.transpose(fuzzy_sets)
    return fuzzy_sets + transpose - np.multiply(fuzzy_sets, transpose)


@nb.njit(fastmath=True, parallel=True)
def low_dimensional_probabilities(distances: np.ndarray, a: float, b: float) -> np.ndarray:
    inv_distances = np.power(1 + a * (distances ** (2 * b)), -1)
    np.fill_diagonal(inv_distances, 0.)
    return inv_distances


@nb.njit(fastmath=True, parallel=True)
def fuzzy_label_intersection(graph: np.ndarray, y: np.array, unknown=1.0, far=2.0):
    n = len(graph)
    for i in range(n):
        for j in range(n):
            if y[i] == -1 and y[j] == -1:
                graph[i][j] *= np.exp(-unknown)
            if y[i] != y[j]:
                graph[i][j] *= np.exp(-far)
    return graph


@nb.experimental.jitclass([
    ('iterations', nb.int32),
    ('m_grad', nb.float64[:,:]),
    ('v_grad', nb.float64[:,:]),
    ('beta1', nb.float32),
    ('beta2', nb.float32),
    ('eta', nb.float32),
])
class Adam():
    def __init__(self, n: int, iterations: int, components=2, eta=1, beta1=0.9, beta2=0.999):
        self.iterations = iterations
        self.m_grad = np.zeros((n, components), dtype=np.float64)
        self.v_grad = np.zeros((n, components), dtype=np.float64)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta

    def step(self, iteration: int, y: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.m_grad = self.beta1 * self.m_grad + (1 - self.beta1) * grad
        self.v_grad = self.v_grad - (1 - self.beta2) * (self.v_grad - grad ** 2)
        return y - self.eta * self.m_grad / (np.sqrt(self.v_grad) + 1e-8)


class GradientOptimizer():
    def __init__(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss

    def run(self, high_probabilities: np.ndarray, y: np.ndarray, a: float, b: float):
        loss_values = []
        step = self.optimizer.step
        for iteration in range(1, self.optimizer.iterations + 1):
            distances = euclidean_distances(y, y)
            grad = self.loss.gradient(high_probabilities, y, distances, a, b)
            loss = self.loss.loss(high_probabilities, low_dimensional_probabilities(distances, a, b))
            loss_values.append(loss)
            if iteration == 1 or iteration % 50 == 0:
                print(f'[{iteration}]: loss {loss}')
            y = step(iteration, y, grad)
        return y, loss_values


def fuzzy_graph(pairwise_distances: np.ndarray, neighbor_counts: List[int]):
    neighbor_counts = np.array(neighbor_counts)
    max_neighbors = np.max(neighbor_counts).item()
    knn_indices, knn_dists = approx_nearest_neighbors(pairwise_distances, max_neighbors)
    fuzzy_sets = local_fuzzy_simplicial_type2_set(len(pairwise_distances), knn_indices, knn_dists, neighbor_counts)
    topological_representation = probabilistic_t_conorm(fuzzy_sets)
    return topological_representation


def umap(
    pairwise_distances: np.ndarray,
    neighbor_counts: List[int],
    n_components: int,
    min_dist: float,
    optimizer: GradientOptimizer,
    y: Union[np.ndarray, None]=None
):
    graph = fuzzy_graph(pairwise_distances, neighbor_counts)
    if y is not None: graph = fuzzy_label_intersection(graph, y)
    y = spectral_embedding(pairwise_distances, graph, n_components)
    a, b = psi_fit(min_dist)
    return optimizer.run(graph, y, a, b)


class CrossEntropyLoss():
    def __init__(self, reduce_repulsion: bool = False):
        self.reduce_repulsion = reduce_repulsion

    def loss(self, high_probabilities: np.ndarray, low_probabilities: np.ndarray):
        left = high_probabilities * np.log(low_probabilities + 1e-8)
        right = np.log(1 - low_probabilities + 1e-8)
        if self.reduce_repulsion:
            right = np.sum(high_probabilities, axis=1) / (2 * len(high_probabilities)) * right
        else:
            right = (1 - high_probabilities) * right
        return -np.sum(left + right)

    def gradient(self, high_probabilities: np.ndarray, y: np.ndarray, distances: np.ndarray, a: float, b: float):
        y_diff = np.expand_dims(y, 1) - np.expand_dims(y, 0)
        inv_dist = 1 / (1 + a * distances ** (2 * b) + 1e-8)
        left = high_probabilities * 2 * a * b * distances ** (2 * b - 1) * inv_dist
        right = (-2) * b * ((distances + 1e-8) ** -1) * inv_dist
        if self.reduce_repulsion:
            right = np.sum(high_probabilities, axis=1) / (2 * len(high_probabilities)) * right
        else:
            right = (1 - high_probabilities) * right
        total = left + right
        summa = np.sum(np.expand_dims(total, 2) * y_diff, axis=1)
        return summa


class SymmetricLoss():
    def __init__(self, reduce_repulsion: bool = False):
        self.reduce_repulsion = reduce_repulsion

    def loss(self, high_probabilities: np.ndarray, low_probabilities: np.ndarray):
        mult = (high_probabilities - low_probabilities)
        top = (high_probabilities * (1 - low_probabilities))
        if self.reduce_repulsion:
            degree = np.sum(high_probabilities, axis=1) / (2 * len(high_probabilities))
            bot = (low_probabilities * degree)
        else:
            bot = (low_probabilities * (1 - high_probabilities))
        log_inner = top / (bot + 1e-8) + 1e-8
        total = mult * np.log(log_inner)
        return np.sum(total)

    def gradient(self, high_probabilities: np.ndarray, y: np.ndarray, distances: np.ndarray, a: float, b: float):
        y_diff = np.expand_dims(y, 1) - np.expand_dims(y, 0)
        ax2b = a * distances ** (2 * b)
        x = distances
        p = high_probabilities
        top_polynom = (ax2b + 1) * (ax2b * p + p - 1)
        if self.reduce_repulsion:
            degree = np.sum(high_probabilities, axis=1) / (2 * len(high_probabilities))
            top_second = ax2b * np.log(degree + 1e-8)
        else:
            top_second = ax2b * np.log(1 - p + 1e-8)
        top_last = ax2b * np.log(ax2b * p + 1e-8)
        top = 2 * b * (top_polynom - top_second + top_last)
        bottom = x * (ax2b + 1) ** 2
        total = top / (bottom + 1e-8)
        summa = np.sum(np.expand_dims(total, 2) * y_diff, axis=1)
        return summa


class IntuitionisticLoss():
    def __init__(self, reduce_repulsion: bool = False):
        self.reduce_repulsion = reduce_repulsion

    def loss(self, high_probabilities: np.ndarray, low_probabilities: np.ndarray):
        half = 0.5 * (high_probabilities + low_probabilities)
        left = high_probabilities * np.log(half + 1e-8)
        if self.reduce_repulsion:
            degree = np.sum(high_probabilities, axis=1) / (2 * len(high_probabilities))
            right = degree * np.log(1 - half + 1e-8)
        else:
            right = (1 - high_probabilities) * np.log(1 - half + 1e-8)
        return -np.sum(left + right)

    def gradient(self, high_probabilities: np.ndarray, y: np.ndarray, distances: np.ndarray, a: float, b: float):
        y_diff = np.expand_dims(y, 1) - np.expand_dims(y, 0)
        x2b = distances ** (2 * b)
        multiplier = 2 * a * b * x2b * ((distances * (1 + a * x2b) + 1e-8) ** -1)
        hpax2b = high_probabilities * a * x2b + high_probabilities
        left = high_probabilities * ((hpax2b + 1) ** -1)
        if self.reduce_repulsion:
            degree = np.sum(high_probabilities, axis=1) / (2 * len(high_probabilities))
            right_mult = degree
        else:
            right_mult = (1 - high_probabilities)
        right = right_mult * ((hpax2b - 2 * a * x2b - 1 + 1e-8) ** -1)
        total = multiplier * (left + right) 
        summa = np.sum(np.expand_dims(total, 2) * y_diff, axis=1)
        return summa