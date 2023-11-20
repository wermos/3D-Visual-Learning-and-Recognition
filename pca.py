import numpy as np

from constants import PCA_THRESHOLD

def PCA(datapoints):
    L = datapoints.T @ datapoints
    eigenvalues, eigenvectors = np.linalg.eig(L)
    num_components = np.searchsorted(np.cumsum(eigenvalues)/sum(eigenvalues), PCA_THRESHOLD, side = "left")+1
    eigenvectors = datapoints @ eigenvectors[:,:num_components]
    eigenvalues = eigenvalues[:num_components]
    return eigenvalues, eigenvectors, num_components