from __future__ import print_function, division
import numpy as np

# R: SVD of (R, restricted to regression SNPs)
# R2: SVD of (R^2, restricted to regression SNPs)
def invert_weights(R, R2, sigma2g, N, x, typed=None, mode='Winv_ahat_h'):
    if typed is None:
        typed = np.isfinite(x.reshape((len(x),-1)).sum(axis=1))

    # trivial weights
    if mode == 'Winv_ahat_I':
        result = x
    # heuristic, with large-N approximation
    elif mode == 'Winv_ahat_hlN':
        U = R['U'][typed,:]; svs=R['svs']
        result = np.full(x.shape, np.nan)
        result[typed] = (U/(svs**2)).dot(U.T.dot(x[typed]))
    # heuristic, no large N approximation, using (R_o)^2 to approximate (R2)_o
    elif mode == 'Winv_ahat_h':
        U = R['U'][typed,:]; svs=R['svs']
        result = np.full(x.shape, np.nan)
        result[typed] = (U/(sigma2g*svs**2+svs/N)).dot(U.T.dot(x[typed]))
    # heuristic, no large N approximation, using R2 instead of R
    elif mode == 'Winv_ahat_h2':
        U = R2['U'][typed,:]; svs=R2['svs']
        result = np.full(x.shape, np.nan)
        result[typed] = (U/(sigma2g*svs+np.sqrt(svs)/N)).dot(U.T.dot(x[typed]))
    # exact
    elif mode == 'Winv_ahat':
        R_ = (R['U'][typed]*R['svs']).dot(R['U'][typed].T)
        R2_ = (R2['U'][typed]*R2['svs']).dot(R2['U'][typed].T)
        W = R_/N + sigma2g*R2_
        U, svs, _ = np.linalg.svd(W)
        k = np.argmax(np.cumsum(svs)/svs.sum() >= 0.95)
        # k = np.argmax(svs[:-1]/svs[1:] >= 1e5)+1
        print(R['U'].shape, R2['U'].shape, 'k=',k)
        U = U[:,:k]; svs = svs[:k]
        result = np.full(x.shape, np.nan)
        result[typed] = (U/svs).dot(U.T.dot(x[typed]))
    return result
