#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import gudhi as gd
import math
import os
import gudhi.representations
import tikzplotlib
import itertools
import persistenceStatistics as ps
import tensorflow as tf

from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import MinMaxScaler
from keras.datasets import cifar10, cifar100, fashion_mnist
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter1d

colorScheme = {
    "black": "#1A1A1D",
    "gray": "#4E4E50",
    "purpur": "#6F2232",
    "brick": "#950740",
    "fire": "#C3073F",
}


def concatenate_landscapes(
    persLandscape1: np.ndarray, persLandscape2: np.ndarray, resolution: int
) -> list:
    """
        **This function concatenates the persistence landscapes according to homology groups.**

        The computation of homology groups requires a certain resolution for each homology class.
        According to this resolution the direct sum of persistence landscapes has to be concatenated
        in a correct manner, such that the persistent homology can be plotted according to the `n`-dimensional
        persistent homology groups.

        + param **persLandscape1**: persistence landscape, type `np.ndarray`.
        + param **persLandscape2**: persistence landscape, type `np.ndarray`.
        + return **concatenatedLandscape**: direct sum of persistence landscapes, type `list`.
    """
    numberPartition = int(len(persLandscape2[0]) / resolution)
    splitLandscape1 = np.split(persLandscape1[0], numberPartition)
    splitLandscape2 = np.split(persLandscape2[0], numberPartition)
    concatenatedLandscape = []

    for i in range(0, numberPartition):
        concatenatedLandscape.append(
             np.fmax(splitLandscape1[i], splitLandscape2[i]))

    return [np.array(concatenatedLandscape).flatten()]


def compute_persistence_landscape(
    data: np.ndarray,
    res: int = 1000,
    persistenceIntervals: int = 1,
    maxAlphaSquare: float = 1e12,
    filtration: str = ["alphaComplex", "vietorisRips", "tangential"],
    maxDimensions: int = 10,
    edgeLength: float = 1,
    plot: bool = False,
    smoothen: bool = False,
    sigma: int = 3,
) -> np.ndarray:
    """
        **A function for computing persistence landscapes for 2D images.**

        This function computes the filtration of a 2D image dataset, the simplicial complex,
        the persistent homology and then returns the persistence landscape as array. It takes
        the resolution of the landscape as parameter, the maximum size for `alphaSquare` and
        options for certain filtrations.

        + param **data**: data set, type `np.ndarray`.
        + param **res**: resolution, default is `1000`, type `int`.
        + param **persistenceIntervals**: interval for persistent homology, default is `1e12`,type `float`.
        + param **maxAlphaSquare**: max. parameter for delaunay expansion, type `float`.
        + param **filtration**: alphaComplex, vietorisRips, cech, delaunay, tangential, type `str`.
        + param **maxDimensions**: only needed for VietorisRips, type `int`.
        + param **edgeLength**: only needed for VietorisRips, type `float`.
        + param **plot**: whether or not to plot, type `bool`.
        + param **smoothen**: whether or not to smoothen the landscapes, type `bool`.
        + param **sigma**: smoothing factor for gaussian mixtures, type `int`.
        + return **landscapeTransformed**: persistence landscape, type `np.ndarray`.
    """

    if filtration == "alphaComplex":
        simComplex = gd.AlphaComplex(points=data).create_simplex_tree(
            max_alpha_square=maxAlphaSquare
        )
    elif filtration == "vietorisRips":
        simComplex = gd.RipsComplex(
            points=data_A_sample, max_edge_length=edgeLength
        ).create_simplex_tree(max_dimension=maxDimensions)
    elif filtration == "tangential":
        simComplex = gd.AlphaComplex(
            points=data, intrinsic_dimension=len(data.shape) - 1
        ).compute_tangential_complex()

    persistenceDiagram = simComplex.persistence()
    landscape = gd.representations.Landscape(resolution=res)
    landscapeTransformed = landscape.fit_transform(
        [simComplex.persistence_intervals_in_dimension(persistenceIntervals)]
    )

    return landscapeTransformed


def compute_mean_persistence_landscapes(
    data: np.ndarray,
    resolution: int = 1000,
    persistenceIntervals: int = 1,
    maxAlphaSquare: float = 1e12,
    filtration: str = ["alphaComplex", "vietorisRips", "tangential"],
    maxDimensions: int = 10,
    edgeLength: float = 0.1,
    plot: bool = False,
    tikzplot: bool = False,
    name: str = "persistenceLandscape",
    smoothen: bool = False,
    sigma: int = 2,
) -> np.ndarray:
    """
        **This function computes mean persistence diagrams over 2D datasets.**

        The functions shows a progress bar of the processed data and takes the direct
        sum of the persistence modules to get a summary of the landscapes of the various
        samples. Further it can be decided whether or not to smoothen the persistence
        landscape by gaussian filter. A plot can be created with `matplotlib` or as
        another option for scientific reporting with `tikzplotlib`, or both.

        Information: The color scheme has 5 colors defined. Thus 5 homology groups can be
        displayed in different colors.

        + param **data**: data set, type `np.ndarray`.
        + param **resolution**: resolution of persistent homology per group, type `int`.
        + param **persistenceIntervals**: intervals for persistence classes, type `int`.
        + param **maxAlphaSquare**: max. parameter for Delaunay expansion, type `float`.
        + param **filtration**: `alphaComplex`, `vietorisRips` or `tangential`, type `str`.
        + param **maxDimensions**: maximal dimension of simplices, type `int`.
        + param **edgeLength**: length of simplex edge, type `float`.
        + param **plot**: whether or not to plot, type `bool`.
        + param **tikzplot**: whether or not to plot as tikz-picture, type `bool`.
        + param **name**: name of the file to be saved, type `str`.
        + param **smoothen**: whether or not to smoothen the landscapes, type `bool`.
        + param **sigma**: smoothing factor for gaussian mixtures, type `int`.
        + return **meanPersistenceLandscape**: mean persistence landscape, type `np.ndarray`.
    """
    dataShape = data.shape
    dataSize = dataShape[0]
    elementSize = len(data[0].flatten())
    reshapedData = data[0].reshape((int(elementSize / 2), 2))

    for i in tqdm(range(0, dataShape[0])):
        if i == 0:
            meanPersistenceLandscape = compute_persistence_landscape(
                reshapedData,
                res=resolution,
                filtration=filtration,
                persistenceIntervals=persistenceIntervals,
                maxAlphaSquare=maxAlphaSquare,
                maxDimensions=maxDimensions,
                edgeLength=edgeLength,
            )
        else:
            reshapedData = data[i].reshape((int(elementSize / 2), 2))
            persistentLandscape = compute_persistence_landscape(
                reshapedData,
                res=resolution,
                filtration=filtration,
                persistenceIntervals=persistenceIntervals,
                maxAlphaSquare=maxAlphaSquare,
                maxDimensions=maxDimensions,
                edgeLength=edgeLength,
            )
            meanPersistenceLandscape = concatenate_landscapes(
                meanPersistenceLandscape, persistentLandscape, resolution
            )

    xaxis = np.arange(resolution)
    numberPartition = int(len(meanPersistenceLandscape[0]) / resolution)
    splittedLandscape = np.split(meanPersistenceLandscape[0], numberPartition)
    dictLength = len(colorScheme)
    keys = list(colorScheme)

    for i in range(0,len(splittedLandscape)):
        maxima = np.sum(np.r_[1, splittedLandscape[i][1:] < splittedLandscape[i][:-1]] & np.r_[splittedLandscape[i][:-1] < splittedLandscape[i][1:], 1])
        print("There are " + str(maxima) + " local maxima for the " + str(i) + "-th homology group.")

    if plot == True:
        for i in range(0, len(splittedLandscape)):
            # Iterate the dict by the current element modulo it's length.
            if smoothen == True:
                plt.fill(
                    xaxis,
                    gaussian_filter1d(splittedLandscape[i], sigma),
                    colorScheme[keys[i%len(keys)]],
                )
            else:
                plt.fill(
                    xaxis,
                    splittedLandscape[i],
                    colorScheme[keys[i%len(keys)]],
                )
        plt.title("Persistence landscape.")
        plt.show()

    elif tikzplot == True:
        for i in range(0, len(splittedLandscape)):
            # Iterate the dict by the current element modulo it's length.
            if smoothen == True:
                plt.fill(
                    xaxis,
                    gaussian_filter1d(splittedLandscape[i], sigma),
                    colorScheme[keys[i%len(keys)]],
                )
            else:
                plt.fill(
                    xaxis,
                    splittedLandscape[i],
                    colorScheme[keys[i%len(keys)]],
                )
        
        plt.title("Persistence landscape.")
        if os.path.exists(os.getcwd() + "/plot") == True:
            tikzplotlib.save("plot/" + name + ".tex")
        else:
            os.mkdir("plot")
            tikzplotlib.save("plot/" + name + ".tex")

    return meanPersistenceLandscape
