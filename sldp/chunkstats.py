from __future__ import print_function, division
import gc
import numpy as np
import pandas as pd
import scipy.stats as st
import ypy.fs as fs

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

    # store SNP indices of begin- and end-points of chunks
    chunkinfo = pd.DataFrame()

    # collapse data within chunks
    chunk_nums = []; chunk_denoms = []
    for n, (i,j) in enumerate(zip(chunkendpoints[:-1], chunkendpoints[1:])):
        ldblock_ind = [l for l in ldblocks.iloc[i:j].index if l in numerators.keys()]
        if len(ldblock_ind) > 0:
            chunk_nums.append(sum(
                [numerators[l] for l in ldblock_ind]))
            chunk_denoms.append(sum(
                [denominators[l] for l in ldblock_ind]))
            chunkinfo = chunkinfo.append({
                'ldblock_begin':min(ldblock_ind),
                'ldblock_end':max(ldblock_ind)+1,
                'chr_begin':ldblocks.loc[min(ldblock_ind),'chr'],
                'chr_end':ldblocks.loc[max(ldblock_ind),'chr'],
                'bp_begin':ldblocks.loc[min(ldblock_ind),'start'],
                'bp_end':ldblocks.loc[max(ldblock_ind),'end'],
                'snpind_begin':ldblocks.loc[min(ldblock_ind),'snpind_begin'],
                'snpind_end':ldblocks.loc[max(ldblock_ind),'snpind_end'],
                'numsnps':sum(ldblocks.loc[ldblock_ind,'M_H'])},
                ignore_index=True)

    ## compute leave-one-out sums
    loonumerators = []; loodenominators = []
    for i in range(len(chunk_nums)):
        loonumerators.append(sum(chunk_nums[:i]+chunk_nums[(i+1):]))
        loodenominators.append(sum(chunk_denoms[:i]+chunk_denoms[(i+1):]))

    return chunk_nums, chunk_denoms, loonumerators, loodenominators, chunkinfo

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
#   marginal annotation
# q is the numerator of the regression after background annots are residualized out
# r is the denominator of the regression after background annots are residualized out
# mux is the vector of coefficients required to residualize the background out of the
#   marginal annotatioin question
# muy is the vector of coefficients required to residualize the backgroud out of the
#   vector of GWAS summary statistics
def residualize(chunk_nums, chunk_denoms, num_background, k):
    q = np.array([num[num_background+k] for num in chunk_nums])
    r = np.array([denom[num_background+k,num_background+k] for denom in chunk_denoms])

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
        r = r - 2*xiaiT.dot(mux) + aiaiT.dot(mux).dot(mux)

    return q, r, mux, muy

# do sign-flipping to get p-value
def signflip(q, T, printmem=True, mode='sum'):
    def mask(a, t):
        a_ = a.copy()
        a_[np.abs(a_) < t] = 0
        return a_

    print('before sign-flipping:', fs.mem(), 'MB')

    if mode == 'sum': # use sum of q as the test statistic
        score = q.sum()
    elif mode == 'medrank': # examine how far the rank of 0 deviates from the 50th percentile
        score = np.searchsorted(np.sort(q), 0)/len(q) - 0.5
    elif mode == 'thresh': # threshold q at some absolute magnitude threshold
        top = np.percentile(np.abs(q), 75)
        print(top)
        ts = np.arange(0, top, top/10)
        q_thresh = np.array([mask(q, t) for t in ts]).T
        q_thresh /= np.linalg.norm(q_thresh, axis=0)
        scores = np.sum(q_thresh, axis=0)
        score = scores[np.argmax(np.abs(scores))]
    else:
        print('ERROR: invalid mode')
        return None

    null = np.zeros(T); current = 0; block = 100000
    while current < len(null):
        s = (-1)**np.random.binomial(1,0.5,size=(block, len(q)))
        if mode == 'sum':
            null[current:current+block] = s.dot(q)
        elif mode == 'medrank':
            null_q = s[:,:]*q[None,:]
            null_q = np.sort(null_q, axis=1)
            null[current:current+block] = \
                    np.array([np.searchsorted(s, 0)/len(s) - 0.5 for s in null_q])
        elif mode == 'thresh':
            null_q_thresh = s[:,:,None]*q_thresh[None,:,:]
            sums = np.sum(null_q_thresh, axis=1)
            null[current:current+block] = np.array([s[np.argmax(np.abs(s))] for s in sums])

        current += block
        p = max(1, ((np.abs(null) >= np.abs(score)).sum())) / float(current)
        del s; gc.collect()
        if p >= 0.01:
            null = null[:current]
            break

    se = np.abs(score)/np.sqrt(st.chi2.isf(p,1))
    del null; gc.collect()
    print('after sign-flipping:', fs.mem(), 'MB. p=', p)

    return p, score/se
