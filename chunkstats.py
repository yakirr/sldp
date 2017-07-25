from __future__ import print_function, division
import gc
import numpy as np
import scipy.stats as st
import pyutils.fs as fs

# make idependent blocks
def collapse_to_chunks(ldblocks, numerators, denominators, numblocks):
    # define endpoints of chunks
    ldblocks.M_H.fillna(0, inplace=True)
    totalM = ldblocks.M_H.sum()
    chunksize = totalM / numblocks
    avgldblocksize = totalM / (ldblocks.M_H != 0).sum()
    chunkendpoints = [0]
    currldblock = 0; currsize = 0
    while currldblock < len(ldblocks):
        while currsize <= max(1,chunksize-avgldblocksize/2) and currldblock < len(ldblocks):
            currsize += ldblocks.iloc[currldblock].M_H
            currldblock += 1
        currsize = 0
        chunkendpoints += [currldblock]

    # collapse data within chunks
    chunk_nums = []; chunk_denoms = []
    for n, (i,j) in enumerate(zip(chunkendpoints[:-1], chunkendpoints[1:])):
        ldblock_ind = [l for l in ldblocks.iloc[i:j].index if l in numerators.keys()]
        if len(ldblock_ind) > 0:
            chunk_nums.append(sum(
                [numerators[l] for l in ldblock_ind]))
            chunk_denoms.append(sum(
                [denominators[l] for l in ldblock_ind]))

    ## compute leave-one-out sums
    loonumerators = []; loodenominators = []
    for i in range(len(chunk_nums)):
        loonumerators.append(sum(chunk_nums[:i]+chunk_nums[(i+1):]))
        loodenominators.append(sum(chunk_denoms[:i]+chunk_denoms[(i+1):]))

    return chunk_nums, chunk_denoms, loonumerators, loodenominators

# compute estimate of effect size
def get_est(num, denom, k, num_background):
    ind = range(num_background) + [num_background+k]
    num = num[ind]
    denom = denom[ind][:,ind]
    try:
        return np.linalg.solve(denom, num)[-1]
    except np.linalg.linalg.LinAlgError:
        return np.nan

# compute standard error of estimate using jackknife
def jackknife_se(est, loonumerators, loodenominators, k, num_background):
    m = np.ones(len(loonumerators))
    theta_notj = [get_est(nu, de, k, num_background)
            for nu, de in zip(loonumerators, loodenominators)]
    g = len(m)
    n = m.sum()
    h = n/m
    theta_J = g*est - ((n-m)/n*theta_notj).sum()
    tau = est*h - (h-1)*theta_notj
    return np.sqrt(np.mean((tau - theta_J)**2/(h-1)))

# residualize the first num_background annotations out of the num_background+k-th
# marginal annotation
def getq(chunk_nums, chunk_denoms, num_background, k):
    q = np.array([num[num_background+k] for num in chunk_nums])

    if num_background > 0:
        num = sum(chunk_nums)
        denom = sum(chunk_denoms)
        ATA = denom[:num_background][:,:num_background]
        ATy = num[:num_background]
        ATx = denom[:num_background][:,num_background+k]
        muy = np.linalg.solve(ATA, ATy)
        mux = np.linalg.solve(ATA, ATx)
        xiaiT = np.array([d[num_background+k,:num_background]
            for d in chunk_denoms])
        yiaiT = np.array([nu[:num_background]
            for nu in chunk_nums])
        aiaiT = np.array([d[:num_background][:,:num_background]
            for d in chunk_denoms])
        q = q - xiaiT.dot(muy) - yiaiT.dot(mux) + aiaiT.dot(muy).dot(mux)

    return q

# do sign-flipping to get p-value
def signflip(q, T, printmem=True):
    print('before sign-flipping:', fs.mem(), 'MB')
    score = q.sum()
    null = np.zeros(T); current = 0; block = 100000
    while current < len(null):
        s = (-1)**np.random.binomial(1,0.5,size=(block, len(q)))
        null[current:current+block] = s.dot(q)
        current += block
        p = ((np.abs(null) >= np.abs(score)).sum()) / float(current)
        del s; gc.collect()

        if p >= 0.01:
            null = null[:current]
            break
    p = ((np.abs(null) >= np.abs(score)).sum()) / float(len(null))
    p = min(max(p,1./float(len(null))), 1)
    se = np.abs(score)/np.sqrt(st.chi2.isf(p,1))

    del null; gc.collect()
    print('after sign-flipping:', fs.mem(), 'MB')

    return p, score/se
