# packages
import sys
import random
from typing import List
import numpy as np
from gudhi.wasserstein import wasserstein_distance
from gudhi.wasserstein.barycenter import lagrangian_barycenter
from src.data_utils.pd_pm_methods import mesh_gen
from PD_subsample.ApproxPH import wass_dist, dist_mat, get_mean_mesr


def kmeans_plusplus(data: list, n_clusters: int, data_type: str, random_state: int, **kwargs) -> List:
    """Performs the k-means++ initialization algorithm to select initial centroids.

    The function initializes centroids by first randomly selecting a starting point,
    then iteratively choosing subsequent points based on their squared distances
    to the nearest existing centroid, weighted by probability.

    Args:
        data (list): List of data points for clustering.
        n_clusters (int): Number of clusters to form.
        data_type (str): Type of data ('diagram' or 'measure').
        random_state (int): Seed for random number generation.
        **kwargs: Additional keyword arguments for distance calculation.

    Returns:
        list: A list containing the initialized centroids.
    """
    # set random seed
    random.seed(random_state)
    # initialise list of centroids and randomly select 1st centroid
    centroids = [random.choice(data)]
    # calculate distances from chosen centroid to all other data points
    while len(centroids) < n_clusters:
        dist = []
        for data_point in data:
            d = sys.maxsize
            for j in range(len(centroids)):
                if data_type == 'diagram':
                    temp_dist = wasserstein_distances(data_point, [centroids[j]])[0]
                if data_type == 'measure':
                    temp_dist = wass_dist(data_point, centroids[j], kwargs['dist_mat'])
                d = min(d, temp_dist)
            dist.append(d)
        # select new centroids
        sum_squared_dist = sum([d ** 2 for d in dist])
        probs = [(d ** 2) / sum_squared_dist for d in dist]
        centroids.append(random.choices(data, weights=probs, k=1)[0])
    return centroids


def not_equal(c1, c2, n_clusters, centroid_type) -> bool:
    """Determines if two centroids are not equal.

    Compares corresponding elements of the centroids. For diagrams, each persistence
    diagram's homology degrees are compared separately. For measures, checks element-wise
    equality with a tolerance for floating-point precision.

    Args:
        c1 (list or array): First centroid to compare.
        c2 (list or array): Second centroid to compare.
        n_clusters (int): Number of clusters.
        centroid_type (str): Type of centroid ('diagram' or 'measure').

    Returns:
        bool: True if centroids are not equal, False otherwise.
    """
    if centroid_type == 'diagram':
        if c1 == [None] * n_clusters or c2 == [None] * n_clusters:
             not_equal_bool = True
        else:
            c1 = [np.concatenate(c).tolist() for c in c1]
            c2 = [np.concatenate(c).tolist() for c in c2]
            not_equal_bool = all([x != y for x, y in zip(c1, c2)])
    if centroid_type == 'measure':
        not_equal_bool = all([not (np.allclose(x, y)) for x, y in zip(c1, c2)])
    return not_equal_bool


def wasserstein_distances(x, centroids) -> List:
    """Calculates the Wasserstein distances between a diagram and the centroids.

    Computes distances separately for each homology degree (0, 1, 2) and sums them.

    Args:
        x (list): A persistence diagram.
        centroids (list): List of centroids.

    Returns:
        list: List of summed Wasserstein distances to each centroid.
    """

    centroid_dists_0 = [wasserstein_distance(x[0], c[0], order=2) for c in centroids]
    centroid_dists_1 = [wasserstein_distance(x[1], c[1], order=2) for c in centroids]
    centroid_dists_2 = [wasserstein_distance(x[2], c[2], order=2) for c in centroids]
    centroid_dists = [sum(d) for d in zip(centroid_dists_0, centroid_dists_1,
                                          centroid_dists_2)]
    return centroid_dists


def get_barycenter(diagrams) -> List:
    """Computes the Fréchet mean of a set of persistence diagrams.

    Splits the diagrams by homology degree and computes the barycenter for each
    part separately.

    Args:
        diagrams (list): List of persistence diagrams.

    Returns:
        list: A list containing the computed means for each homology degree.
    """
    diagrams_0 = []
    diagrams_1 = []
    diagrams_2 = []
    # concatenate subdiagrams into single list for comparison
    for diag in diagrams:
        diagrams_0.append(diag[0])
        diagrams_1.append(diag[1])
        diagrams_2.append(diag[2])
    # compute Fréchet mean of subdiagrams
    fmean_0 = lagrangian_barycenter(diagrams_0, init=0, verbose=False)
    fmean_1 = lagrangian_barycenter(diagrams_1, init=0, verbose=False)
    fmean_2 = lagrangian_barycenter(diagrams_2, init=0, verbose=False)
    # return whole diagram
    return [fmean_0, fmean_1, fmean_2]


class PD_KMeans:
    """A k-means clustering class for persistence diagrams.

    Attributes:
        n_clusters (int): Number of clusters.
        init (str): Initialization method ('kmeans++' or 'random').
        max_iters (int): Maximum number of iterations.
        random_state (int): Seed for random number generation.
    """
    def __init__(self, n_clusters: int, init: str ='kmeans++', random_state: int = 1245, max_iters: int = 25):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iters = max_iters
        self.random_state = random_state

    def fit(self, diagrams: list) -> List:
        """Fits the k-means clustering to the given persistence diagrams.

        Args:
            diagrams (list): List of persistence diagrams.

        Returns:
            list: A list of cluster labels for each diagram.
        """
        # set parameters
        random.seed(self.random_state)
        iteration = 0
        prev_centroids = [None] * self.n_clusters

        # initialisation step
        if self.init == 'random':
            self.centroids = random.sample(diagrams, self.n_clusters)
        if self.init == 'kmeans++':
            self.centroids = kmeans_plusplus(
                diagrams,
                n_clusters=self.n_clusters,
                data_type='diagram',
                random_state=self.random_state
            )

        while not_equal(prev_centroids, self.centroids, n_clusters=self.n_clusters, centroid_type='diagram') \
                and iteration < self.max_iters:
            clusters = [[] for _ in range(self.n_clusters)]
            labels = []
            # assignment step
            for diag in diagrams:
                dists = wasserstein_distances(diag, self.centroids)
                assigned_centroid = np.argmin(dists)
                clusters[assigned_centroid].append(diag)
                labels.append(assigned_centroid)

            # update step
            prev_centroids = self.centroids
            self.centroids = [get_barycenter(cluster) for cluster in clusters]

            # increase iteration
            iteration += 1

        return labels


class PM_KMeans:
    """A k-means clustering class for persistence measures.

    Attributes:
        n_clusters (int): Number of clusters.
        grid_width (int): Grid width for measure generation.
        init (str): Initialization method ('kmeans++' or 'random').
        random_state (int): Seed for random number generation.
        max_iters (int): Maximum number of iterations.
    """
    def __init__(self, n_clusters: int, grid_width : int, init : str = 'kmeans++', random_state : int = 1245, max_iters: int = 25):
        self.n_clusters = n_clusters
        self.grid_width = grid_width
        self.init = init
        self.random_state = random_state
        self.max_iters = max_iters

    def fit(self, measures: list) -> List:
        """Fits the k-means clustering to the given persistence measures.

        Args:
            measures (list): List of persistence measures.

        Returns:
            list: A list of cluster labels for each measure.
        """
        # set parameters
        random.seed(self.random_state)
        iteration = 0
        labels = []
        prev_centroids = [np.zeros(measures[0].shape) for _ in range(self.n_clusters)]
        grid = mesh_gen(self.grid_width)

        self.mp = dist_mat(grid, 2)

        if self.init == 'random':
            self.centroids = random.sample(measures, self.n_clusters)
        if self.init == 'kmeans++':
            self.centroids = kmeans_plusplus(
                measures,
                n_clusters=self.n_clusters,
                data_type='measure',
                random_state=self.random_state,
                dist_mat = self.mp
            )

        # initialisation step
        self.centroids = random.sample(measures, self.n_clusters)

        while not_equal(prev_centroids, self.centroids, n_clusters=self.n_clusters, centroid_type='measure') and (iteration < self.max_iters) :

            # reset
            clusters = [[] for _ in range(self.n_clusters)]
            labels = []

            # assignment step
            for mesr in measures:
                dists = [wass_dist(centroid, mesr, self.mp) for centroid in self.centroids]
                assigned_centroid = np.argmin(dists)
                clusters[assigned_centroid].append(mesr)
                labels.append(assigned_centroid)
            if len(set(labels)) != 3:
                break

            # update step
            prev_centroids = self.centroids
            self.centroids = [get_mean_mesr(cluster, float_error=1e-8) for cluster in clusters]

            # increase iteration
            iteration += 1

        return labels
