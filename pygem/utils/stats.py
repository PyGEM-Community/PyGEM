"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distrubted under the MIT lisence

Model statistics module
"""

import numpy as np
import arviz as az

def effective_n(x):
    """
    Compute the effective sample size of a trace.

    Takes the trace and computes the effective sample size
    according to its detrended autocorrelation.

    Parameters
    ----------
    x : list or array of chain samples

    Returns
    -------
    effective_n : int
        effective sample size
    """
    if len(set(x)) == 1:
        return 1
    try:
        # detrend trace using mean to be consistent with statistics
        # definition of autocorrelation
        x = np.asarray(x)
        x = (x - x.mean())
        # compute autocorrelation (note: only need second half since
        # they are symmetric)
        rho = np.correlate(x, x, mode='full')
        rho = rho[len(rho)//2:]
        # normalize the autocorrelation values
        #  note: rho[0] is the variance * n_samples, so this is consistent
        #  with the statistics definition of autocorrelation on wikipedia
        # (dividing by n_samples gives you the expected value).
        rho_norm = rho / rho[0]
        # Iterate until sum of consecutive estimates of autocorrelation is
        # negative to avoid issues with the sum being -0.5, which returns an
        # effective_n of infinity
        negative_autocorr = False
        t = 1
        n = len(x)
        while not negative_autocorr and (t < n):
            if not t % 2:
                negative_autocorr = sum(rho_norm[t-1:t+1]) < 0
            t += 1
        return int(n / (1 + 2*rho_norm[1:t].sum()))
    except:
        return None
    

def mcmc_stats(chains_dict, 
               params=['tbias','kp','ddfsnow','ddfice','rhoabl', 'rhoacc','mb_mwea']):
    """
    Compute per-chain and overall summary stats for MCMC samples.

    Parameters
    ----------
    chains_dict : dict
        Dictionary with structure:
        {
            "param1": {
                "chain1": [...],
                "chain2": [...],
                ...
            },
            ...
        }

    Returns
    -------
    summary_stats : dict
        Dictionary with structure:
        {
            "param1": {
                "mean": [...],         # per chain
                "std": [...],
                "median": [...],
                "q025": [...],
                "q975": [...],
                "ess": ...,            # overall
                "r_hat": ...           # overall
            },
            ...
        }
    """
    summary_stats = {}

    for param, chains in chains_dict.items():
        if param not in params:
            continue

        # Stack chains into array: shape (n_chains, n_samples)
        chain_names = sorted(chains)  # ensure consistent order
        samples = np.array([chains[c] for c in chain_names])

        # Per-chain stats
        means = np.mean(samples, axis=1).tolist()
        stds = np.std(samples, axis=1, ddof=1).tolist()
        medians = np.median(samples, axis=1).tolist()
        q25 = np.quantile(samples, 0.25, axis=1).tolist()
        q75 = np.quantile(samples, 0.75, axis=1).tolist()
        ess = [effective_n(x) for x in samples]
        # Overall stats (R-hat)
        if samples.shape[0] > 1:
            # calculate the gelman-rubin stat for each variable across all chains
            # pass chains as 2d array to arviz using the from_dict() method
            # convert the chains into an InferenceData object
            idata = az.from_dict(posterior={param: samples})
            # calculate the Gelman-Rubin statistic (rhat)
            r_hat = float(az.rhat(idata).to_array().values[0])
        else :
            r_hat = None

        summary_stats[param] = {
            "mean": means,
            "std": stds,
            "median": medians,
            "q25": q25,
            "q75": q75,
            "ess": ess,
            "r_hat": r_hat
        }

    chains_dict['_summary_stats_'] = summary_stats

    return chains_dict