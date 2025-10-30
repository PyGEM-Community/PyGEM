"""
Python Glacier Evolution Model (PyGEM)

copyright © 2025 Brandon Tober <btober@cmu.edu>, David Rounce <drounce@cmu.edu>

Distributed under the MIT license

Markov chain Monte Carlo methods
"""

import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from pygem.setup.config import ConfigManager

# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()

torch.set_default_dtype(torch.float64)
plt.rcParams['font.family'] = 'arial'
plt.rcParams['font.size'] = 8
plt.rcParams['legend.fontsize'] = 6


# z-normalization functions
def z_normalize(params, means, std_devs):
    return (params - means) / std_devs


# inverse z-normalization
def inverse_z_normalize(z_params, means, std_devs):
    return z_params * std_devs + means


def log_normal_density(x, method='mean', **kwargs):
    """
    Evaluate the log probability density of a normal distribution.

    Parameters:
    - x: input data point or array of data points.
    - mu: mean of the normal distribution (diagonal covariance matrix).
    - sigma: standard deviation (diagonal elements of the covariance matrix).

    Returns:
        log probability density
    """
    mu, sigma = kwargs['mu'], kwargs['sigma']

    # apply different uncertainty if prediction is above and below the observation
    if sigma.ndim == 2 and sigma.shape[0] == 2:
        # two-sided uncertainty: use the observation uncertainty that is on the same side as the prediction
        sigma_min, sigma_max = sigma
        sigma = torch.where(mu > x, sigma_max, sigma_min)

    # ensure tensors are flattened
    x, mu, sigma = map(torch.flatten, (x, mu, sigma))

    # compute log normal density per element
    k = mu.shape[-1]

    # scale sigma by sqrt(k)
    # sigma *= torch.sqrt(torch.tensor(k))

    # compute log normal density per element
    log_prob = -k / 2.0 * torch.log(torch.tensor(2 * np.pi)) - torch.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2

    if method == 'sum':
        return torch.tensor([log_prob.nansum()])
    elif method == 'mean':
        return torch.tensor([log_prob.nanmean()])
    elif method == 'weighted':
        # weight each observation
            # obs_weight=1: each observation has full, equal weight; equivalend to method='sum'
            # obs_weight=0: observations have minimum weight; equivalent to method='mean'
        obs_weight = kwargs.get('obs_weight', 0.0)
        obs_n = torch.sum(~torch.isnan(log_prob))
        obs_n_weighted = 1 + (obs_n - 1) * (1 - obs_weight)
        return torch.tensor([log_prob.nansum() / obs_n_weighted])
    else:
        raise ValueError("method must be one of ['sum', 'mean', 'weighted']")


def log_gamma_density(x, **kwargs):
    """
    Computes the log probability density of a Gamma distribution.

    Parameters:
    - x: Input tensor where you want to evaluate the log probability.
    - alpha: Shape parameter of the Gamma distribution.
    - beta: Rate parameter (1/scale) of the Gamma distribution.

    Returns:
        Log probability density at the given input tensor x.
    """
    alpha, beta = kwargs['alpha'], kwargs['beta']  # shape, scale
    return alpha * torch.log(beta) + (alpha - 1) * torch.log(x) - beta * x - torch.lgamma(alpha)


def log_truncated_normal_density(x, **kwargs):
    """
    Computes the log probability density of a truncated normal distribution.

    Parameters:
    - x: Input tensor where you want to evaluate the log probability.
    - mu: Mean of the normal distribution.
    - sigma: Standard deviation of the normal distribution.
    - a: Lower truncation bound.
    - b: Upper truncation bound.

    Returns:
        Log probability density at the given input tensor x.
    """
    mu, sigma, lo, hi = kwargs['mu'], kwargs['sigma'], kwargs['low'], kwargs['high']
    # Standardize
    standard_x = (x - mu) / sigma
    standard_a = (lo - mu) / sigma
    standard_b = (hi - mu) / sigma

    # PDF of the standard normal distribution
    pdf = torch.exp(-0.5 * standard_x**2) / np.sqrt(2 * torch.pi)

    # CDF of the standard normal distribution using the error function
    cdf_upper = 0.5 * (1 + torch.erf(standard_b / np.sqrt(2)))
    cdf_lower = 0.5 * (1 + torch.erf(standard_a / np.sqrt(2)))

    normalization = cdf_upper - cdf_lower

    return torch.log(pdf) - torch.log(normalization)


def log_uniform_density(x, **kwargs):
    """
    Computes the log probability density of a Uniform distribution for scalar x.

    Parameters:
    - x: Scalar tensor where you want to evaluate the log probability.
    - low: Lower bound of the uniform distribution.
    - high: Upper bound of the uniform distribution.

    Returns:
        Scalar log probability density at x.
    """
    low, high = kwargs['low'], kwargs['high']
    if low <= x <= high:
        return -torch.log(high - low)
    else:
        return torch.tensor([float('-inf')])


# mapper dictionary - maps to appropriate log probability density function for given distribution `type`
log_prob_fxn_map = {
    'normal': log_normal_density,
    'gamma': log_gamma_density,
    'truncnormal': log_truncated_normal_density,
    'uniform': log_uniform_density,
}


# mass balance posterior class
class mbPosterior:
    def __init__(self, obs, priors, fxn2eval, fxnargs=None, potential_fxns=None, **kwargs):
        # obs will be passed as a list, where each item is a tuple with the first element being the mean observation, and the second being the variance
        self.obs = obs
        self.priors = copy.deepcopy(priors)
        self.fxn2eval = fxn2eval
        self.fxnargs = fxnargs
        self.potential_functions = potential_fxns if potential_fxns is not None else []
        self.preds = None
        self.check_priors()

        self.ela = kwargs.get('ela', None)
        self.bin_z = kwargs.get('bin_z', None)
        if self.ela:
            self.abl_mask = self.bin_z < self.ela

        # get mean and std for each parameter type
        self.means = torch.tensor([params['mu'] for params in self.priors.values()])
        self.stds = torch.tensor([params['sigma'] for params in self.priors.values()])

    # check priors. remove any subkeys that have a `None` value, and ensure that we have a mean and standard deviation for and gamma distributions
    def check_priors(self):
        for k in list(self.priors.keys()):
            keys_rm = []  # List to hold keys to remove
            for i, value in self.priors[k].items():
                if value is None:
                    keys_rm.append(i)  # Add key to remove list
                # ensure torch tensor objects
                elif isinstance(value, str) and 'inf' in value:
                    self.priors[k][i] = torch.tensor([float(value)])
                elif isinstance(value, float):
                    self.priors[k][i] = torch.tensor([self.priors[k][i]])
            # Remove the keys outside of the iteration
            for i in keys_rm:
                del self.priors[k][i]

        for k in self.priors.keys():
            if self.priors[k]['type'] == 'gamma' and 'mu' not in self.priors[k].keys():
                self.priors[k]['mu'] = self.priors[k]['alpha'] / self.priors[k]['beta']
                self.priors[k]['sigma'] = float(np.sqrt(self.priors[k]['alpha']) / self.priors[k]['beta'])

            if self.priors[k]['type'] == 'uniform' and 'mu' not in self.priors[k].keys():
                self.priors[k]['mu'] = (self.priors[k]['low'] / self.priors[k]['high']) / 2
                self.priors[k]['sigma'] = (self.priors[k]['high'] - self.priors[k]['low']) / (12 ** (1 / 2))

    # update modelprms for evaluation
    def update_modelprms(self, m):
        for i, k in enumerate(['tbias', 'kp', 'ddfsnow']):
            self.fxnargs[1][k] = float(m[i])
        self.fxnargs[1]['ddfice'] = self.fxnargs[1]['ddfsnow'] / pygem_prms['sim']['params']['ddfsnow_iceratio']

    # get model predictions
    def get_model_pred(self, m):
        self.update_modelprms(m)  # update modelprms with current step
        self.preds = self.fxn2eval(*self.fxnargs)
        # convert all values to torch tensors
        self.preds = {k: torch.tensor(v, dtype=torch.float) for k, v in self.preds.items()}

    # get total log prior density
    def log_prior(self, m):
        log_prior = []
        for i, (key, params) in enumerate(self.priors.items()):
            params_copy = params.copy()
            prior_type = params_copy.pop('type')
            function_to_call = log_prob_fxn_map[prior_type]
            log_prior.append(function_to_call(m[i], **params_copy))
        log_prior = torch.stack(log_prior).sum()
        return log_prior

    # get log likelihood
    def log_likelihood(self, m):
        log_likehood = 0
        for k, pred in self.preds.items():
            # --- Check for invalid predictions  ---
            if torch.all(pred == float('-inf')):
                # Invalid model output -> assign -inf likelihood
                return torch.tensor([-float('inf')])

            # if key is `elev_change_1d` scale by density to predict binned surface elevation change
            if k == 'elev_change_1d':
                # Create density field, separate values for ablation/accumulation zones
                rho = np.ones_like(self.bin_z)
                rho[self.abl_mask] = m[3]  # rhoabl
                rho[~self.abl_mask] = m[4]  # rhoacc
                rho = torch.tensor(rho)
                # scale prediction by model density values (convert from m ice to m surface elevation change considering modeled density)
                pred *= pygem_prms['constants']['density_ice'] / rho[:, np.newaxis]
                # update values in preds dict
                self.preds[k] = pred

            log_likehood += log_normal_density(
                self.obs[k][0],  # observations
                mu=pred,  # scaled predictions
                sigma=self.obs[k][1],  # uncertainty
            )
            # log_likehood += log_normal_density(
            #     self.obs[k][0],  # observed values
            #     method='weighted', # use weighted observations
            #     obs_weight=1, # observation weight
            #     mu=pred,  # predicted values
            #     sigma=self.obs[k][1],  # observation uncertainty
            # )

        return log_likehood

    # compute the log-potential, summing over all declared potential functions.
    def log_potential(self, m):
        # --- Base arguments ---
        # kp, tbias, ddfsnow, massbal
        kwargs = {
            'kp': m[0],
            'tbias': m[1],
            'ddfsnow': m[2],
            'massbal': self.preds['glacierwide_mb_mwea'],
        }

        # --- Optional arguments(if len(m) > 3) ---
        # rhoabl, rhoacc
        if len(m) > 3:
            kwargs['rhoabl'] = m[-2]
            kwargs['rhoacc'] = m[-1]

        # --- Evaluate all potential functions ---
        return sum(pf(**kwargs) for pf in self.potential_functions)

    # get log posterior (sum of log prior, log likelihood and log potential)
    def log_posterior(self, m):
        # anytime log_posterior is called for a new step, calculate the predicted mass balance
        self.get_model_pred(m)
        return self.log_prior(m) + self.log_likelihood(m) + self.log_potential(m), self.preds


# Metropolis-Hastings Markov chain Monte Carlo class
class Metropolis:
    def __init__(self, means, stds):
        # Initialize chains
        self.steps = []
        self.P_chain = []
        self.m_chain = []
        self.m_primes = []
        self.preds_chain = {}
        self.preds_primes = {}
        self.naccept = 0
        self.acceptance = []
        self.n_rm = 0
        self.means = means
        self.stds = stds

    def get_n_rm(self, tol=0.1):
        """
        get the number of samples from the beginning of the chain where the sampler is stuck
        Parameters:
        tol: float representing the tolerance in z-normalized space
        """
        n_params = len(self.m_chain[0])
        n_rms = []
        # get z-normalized vals
        z_norms = [z_normalize(vals, self.means, self.stds) for vals in self.m_chain]
        for i in range(n_params):
            param_vals = [vals[i] for vals in z_norms]
            first_value = param_vals[0]
            count = 0
            for value in param_vals:
                if abs(value - first_value) <= tol:
                    count += 1
                else:
                    break  # Stop counting when we find a value outside the tolerance
            n_rms.append(count)
        self.n_rm = max(n_rms)
        return

    def rm_stuck_samples(self):
        """
        remove stuck samples at the beginning of the chain
        """
        self.P_chain = self.P_chain[self.n_rm :]
        self.m_chain = self.m_chain[self.n_rm :]
        self.m_primes = self.m_primes[self.n_rm :]
        self.steps = self.steps[self.n_rm :]
        self.acceptance = self.acceptance[self.n_rm :]
        for k in self.preds_primes.keys():
            self.preds_primes[k] = self.preds_primes[k][self.n_rm :]
            self.preds_chain[k] = self.preds_chain[k][self.n_rm :]
        return

    def sample(
        self,
        m_0,
        log_posterior,
        n_samples=1000,
        h=0.1,
        burnin=0,
        thin_factor=1,
        trim=True,
        progress_bar=False,
    ):
        # Compute initial unscaled log-posterior
        P_0, pred_0 = log_posterior(m_0)

        n = len(m_0)

        # Create a tqdm progress bar if enabled
        pbar = tqdm(total=n_samples) if progress_bar else None

        i = 0
        # Draw samples
        while i < n_samples:
            # Propose new value according to
            # proposal distribution Q(m) = N(m_0,h)
            step = torch.randn(n) * h

            # update m_prime based on normalized values
            m_prime = z_normalize(m_0, self.means, self.stds) + step
            m_prime = inverse_z_normalize(m_prime, self.means, self.stds)

            # Compute new unscaled log-posterior
            P_1, pred_1 = log_posterior(m_prime)

            # Compute logarithm of probability ratio
            log_ratio = P_1 - P_0

            # Convert to non-log space
            ratio = torch.exp(log_ratio)

            # If proposed value is more probable than current value, accept.
            # If not, then accept proportional to the probability ratios
            if ratio > torch.rand(1):
                m_0 = m_prime
                P_0 = P_1
                pred_0 = pred_1
                # update naccept
                self.naccept += 1

            # Only append to the chain if we're past burn-in.
            if i > burnin:
                # Only append every j-th sample to the chain
                if i % thin_factor == 0:
                    self.steps.append(step)
                    self.P_chain.append(P_0)
                    self.m_chain.append(m_0)
                    self.m_primes.append(m_prime)
                    self.acceptance.append(self.naccept / (i + (thin_factor * self.n_rm)))
                    for k, values in pred_1.items():
                        if k not in self.preds_chain.keys():
                            self.preds_chain[k] = []
                            self.preds_primes[k] = []
                        self.preds_chain[k].append(pred_0[k])
                        self.preds_primes[k].append(pred_1[k])

            # trim off any initial steps that are stagnant
            if (i == (n_samples - 1)) and (trim):
                self.get_n_rm()
                if self.n_rm > 0:
                    if self.n_rm < len(self.m_chain) - 1:
                        self.rm_stuck_samples()
                        i -= int((self.n_rm) * thin_factor)  # back track the iterator
                    trim = False  # set trim to False as to only perform one time

            # increment iterator
            i += 1

            # update progress bar
            if pbar:
                pbar.update(1)

        # Close the progress bar if it was used
        if pbar:
            pbar.close()

        return (
            torch.vstack(self.m_chain),
            self.preds_chain,
            torch.vstack(self.m_primes),
            self.preds_primes,
            torch.vstack(self.steps),
            self.acceptance,
        )
