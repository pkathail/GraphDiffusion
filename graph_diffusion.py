import numpy as np
import pandas as pd
import time
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, find

def run_diffusion_map(scdata, knn=10, epsilon=1, 
        n_diffusion_components=10, n_pca_components=15, markers=None, normalization='markov'):
    """ Run diffusion maps on the data. Run on the principal component projections
    for single cell RNA-seq data and on the expression matrix for mass cytometry data
    :param knn: Number of neighbors for graph construction to determine distances between cells
    :param epsilon: Gaussian standard deviation for converting distances to affinities
    :param n_diffusion_components: Number of diffusion components to Generalte
    :param n_pca_components: Number of components to use for running tSNE for single cell 
    RNA-seq data. Ignored for mass cytometry
    :return: None
    """

    data = deepcopy(scdata)
    if scdata.data_type == 'sc-seq':
        if scdata.pca is None:
            raise RuntimeError('Please run PCA using run_pca before running diffusion maps for single cell RNA-seq')

        data = deepcopy(scdata)
        data -= np.min(np.ravel(data))
        data /= np.max(np.ravel(data))
        data = pd.DataFrame(np.dot(data, scdata.pca['loadings'].iloc[:, 0:n_pca_components]),
                            index=data.index)

    if markers is None:
        markers = scdata.columns

    if data.data_type == 'masscyt':
        data = deepcopy(scdata[markers])

    # Nearest neighbors
    N = data.shape[0]
    nbrs = NearestNeighbors(n_neighbors=knn).fit(data)
    distances, indices = nbrs.kneighbors(data)

    # Adjacency matrix
    rows = np.zeros(N * knn, dtype=np.int32)
    cols = np.zeros(N * knn, dtype=np.int32)
    dists = np.zeros(N * knn)
    location = 0
    for i in range(N):
        inds = range(location, location + knn)
        rows[inds] = indices[i, :]
        cols[inds] = i
        dists[inds] = distances[i, :]
        location += knn
    W = csr_matrix( (dists, (rows, cols)), shape=[N, N] )

    # Symmetrize W
    W = W + W.T

    # Convert to affinity (with selfloops)
    rows, cols, dists = find(W)
    rows = np.append(rows, range(N))
    cols = np.append(cols, range(N))
    dists = np.append(dists/(epsilon ** 2), np.zeros(N))
    W = csr_matrix( (np.exp(-dists), (rows, cols)), shape=[N, N] )

    # Create D
    D = np.ravel(W.sum(axis = 1))
    D[D!=0] = 1/D[D!=0]

    #Go through the various normalizations
    start = time.process_time()
	if normalization == 'bimarkov':
    	# T = Bimarkov(W);
	elif normalization == 'smarkov':
    	print('(symmetric markov) ... ')

    	D = csr_matrix((np.sqrt(D), (range(N), range(N))),  shape=[N, N])
    	P = D
    	T = D.dot(W).dot(D)

    	T = (T + T.T) / 2
    
	elif normalization == 'markov':
    	print('(markov) ... ')

    	T = csr_matrix((D, (range(N), range(N))), shape=[N, N]).dot(W)
    
	elif normalization == 'sbeltrami':
    	print('(symmetric beltrami) ... ')
    
    	P = csr_matrix((D, (range(N), range(N))), shape=[N, N])
    	K = P.dot(W).dot(P)
    
    	D = np.ravel(K.sum(axis = 1))
    	D[D!=0] = 1/D[D!=0]
    
    	D = csr_matrix((D, (range(N), range(N))),  shape=[N, N])
    	P = D
    	T = D.dot(K).dot(D)

    	T = (T + T.T) / 2	# iron out numerical wrinkles
    
	elif normalization == 'beltrami':
    	print('(beltrami) ... ')

    	D = csr_matrix((D, (range(N), range(N))), shape=[N, N])
    	K = D.dot(W).dot(D)
    
    	D = np.ravel(K.sum(axis = 1))
    	D[D!=0] = 1/D[D!=0]
    
    	V = csr_matrix((D, (range(N), range(N))), shape=[N, N])
    	T = V.dot(K)
    
	elif normalization == 'FokkerPlanck':
    	print('(FokkerPlanck) ... ')
    
    	D = csr_matrix((np.sqrt(D), (range(N), range(N))),  shape=[N, N])
    	K = D.dot(W).dot(D)
    
    	D = np.ravel(K.sum(axis = 1))
    	D[D!=0] = 1/D[D!=0]
    
    	D = csr_matrix((D, (range(N), range(N))), shape=[N, N])
    	T = D.dot(K)
    
	elif normalization == 'sFokkerPlanck':
    	print('(sFokkerPlanck) ... ')

    	D = csr_matrix((np.sqrt(D), (range(N), range(N))),  shape=[N, N])
    	K = D.dot(W).dot(D)
    
    	D = np.ravel(K.sum(axis = 1))
    	D[D!=0] = 1/D[D!=0]
    
    	D = csr_matrix((np.sqrt(D), (range(N), range(N))),  shape=[N, N])
    	P = D
    	T = D.dot(K).dot(D)
    
    	T = (T + T.T) / 2

	else:
    	print('\nGraphDiffusion:Warning: unknown normalization.')
    	return

	if normalization != 'bimarkov':
    	print('%.2f seconds' % (time.process_time()-start))

    # Eigen value decomposition
    D, V = eigs(T, n_diffusion_components, tol=1e-4, maxiter=1000)
    D = np.real(D)
    V = np.real(V)
    inds = np.argsort(D)[::-1]
    D = D[inds]
    V = V[:, inds]
    V = P.dot(V)

    # Normalize
    for i in range(V.shape[1]):
        V[:, i] = V[:, i] / norm(V[:, i])
    V = np.round(V, 10)

    # Update object
    scdata.diffusion_eigenvectors = pd.DataFrame(V, index=scdata.data.index)
    scdata.diffusion_eigenvalues = pd.DataFrame(D)