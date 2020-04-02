#!/usr/bin/env python

import numpy as np
import typing

from multiprocessing import Pool
from sklearn.neighbors import KDTree


def hausd_interval(
    data: np.ndarray,
    confidenceLevel: float = 0.95,
    subsampleSize: int = -1,
    subsampleNumber: int = 1000,
    pairwiseDist: bool = False,
    leafSize: int = 2,
    ncores: int = 2,
) -> float:
    """
        **Computation of Hausdorff distance based confidence values.**

        Measures the confidence between two persistent features, wether they are drawn from
        a distribution fitting the underlying manifold of the data. This function is based on
        the Hausdorff distance between the points.

        + param **data**: a data set, type `np.ndarray`.
        + param **confidenceLevel**: confidence level, default `0.95`, type `float`.
        + param **subsampleSize**: size of each subsample, type `int`.
        + param **subsampleNumber**: number of subsamples, type `int`.
        + param **pairwiseDist**: if `true`, a symmetric `nxn`-matrix is generated out of the data, type `bool`.
        + param **leafSize**: leaf size for KDTree, type `int`.
        + param **ncores**: number of cores for parallel computing, type `int`.
        + return **confidence**: the confidence to be a persistent homology class, type `float`.
    """
    dataSize = np.size(data, 0)

    if subsampleSize == -1:
        subsampleSize = int(dataSize / np.log(dataSize))
        global hausdorff_distance

    if pairwiseDist == False:

        def hausdorff_distance(subsampleSize: list) -> float:
            """
                **Distances between the points of data and a random subsample of data of size `m`.**

                + param **subsampleSize**: the size of the data, type `int`.
                + return **hausdorffDistance**: Hausdorff distance, type `float`.
            """
            I = np.random.choice(dataSize, subsampleSize)
            Icomp = [item for item in np.arange(dataSize) if item not in I]
            tree = KDTree(data[I,], leaf_size=leafSize)
            distance, ind = tree.query(data[Icomp,], k=1)
            hausdorffDistance = max(distance)
            return hausdorffDistance

        with Pool(ncores) as cores:
            distanceVector = cores.map(
                hausdorff_distance, [subsampleSize] * subsampleNumber
            )
        cores.close()

    else:

        def hausdorff_distance(subsampleSize: list) -> float:
            """
                **Distances between the points of data and a random subsample of data of size `m`.**

                + param **subsampleSize**: the size of the data, type `int`.
                + return **hausdorffDistance**: Hausdorff distance, type `float`.
            """
            I = np.random.choice(dataSize, subsampleSize)
            hausdorffDistance = np.max(
                [np.min(data[I, j]) for j in np.arange(dataSize) if j not in I]
            )
            return hausdorffDistance

        with Pool(ncores) as cores:
            distanceVector = cores.map(
                hausdorff_distance, [subsampleSize] * subsampleNumber
            )
        cores.close()
        distanceVector = [i[0] for i in distanceVector]

    # Quantile and confidence band.
    myquantile = np.quantile(distanceVector, confidenceLevel)
    confidence = 2 * myquantile

    return confidence


def truncated_simplex_tree(simplexTree: np.ndarray, int_trunc: int = 100) -> tuple:
    """
        **This function return a truncated simplex tree.**

        A sparse representation of the persistence diagram in the form of a truncated
        persistence tree. Speeds up computation on large scale data sets.

        + param **simplexTree**: simplex tree, type `np.ndarray`.
        + param **int_trunc**: number of persistent interval kept per dimension, default is `100`, type `int`.
        + return **simplexTreeTruncatedPersistence**: truncated simplex tree, type `np.ndarray`.
    """
    simplexTree.persistence()
    dimension = simplexTree.dimension()
    simplexTreeTruncatedPersistence = []

    for i in range(dimension):
        dPersistence = simplexTree.persistence_intervals_in_dimension(dimension)
        j = len(dPersistence)

        if j > int_trunc:
            dPersistenceTruncated = [dPersistence[i] for i in range(j - int_trunc, j)]
        else:
            dPersistenceTruncated = dPersistence
        simplexTreeTruncatedPersistence = simplexTreeTruncatedPersistence + [
            (i, (l[0], l[1])) for l in dPersistenceTruncated
        ]

    return simplexTreeTruncatedPersistence
