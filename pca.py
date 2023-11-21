import numpy as np

import constants

def PCA(datapoints):
    mean = np.mean(datapoints, axis = 1)
    datapoints = datapoints - mean.reshape((constants.IMAGE_SIZE,1))
    L = datapoints.T @ datapoints
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    num_components = np.searchsorted(np.cumsum(eigenvalues)/sum(eigenvalues), constants.PCA_THRESHOLD, side = "left")+1
    eigenvectors = datapoints @ eigenvectors[:,:num_components]
    eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis = 0)
    eigenvalues = eigenvalues[:num_components]
    return eigenvalues, eigenvectors, mean