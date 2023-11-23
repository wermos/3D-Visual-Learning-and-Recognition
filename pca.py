from os import environ
environ['OMP_NUM_THREADS'] = '16'
import numpy as np

def PCA(datapoints, img_size, pca_threshold):
    mean = np.mean(datapoints, axis = 1)
    datapoints = datapoints - mean.reshape((img_size,1))
    L = datapoints.T @ datapoints
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    num_components = np.searchsorted(np.cumsum(eigenvalues)/sum(eigenvalues), pca_threshold, side = "left")+1
    eigenvectors = datapoints @ eigenvectors[:,:num_components]
    eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis = 0)
    eigenvalues = eigenvalues[:num_components]
    return eigenvalues, eigenvectors, mean