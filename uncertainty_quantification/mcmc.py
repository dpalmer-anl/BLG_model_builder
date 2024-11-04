import warnings
from typing import Callable, List, Optional

import numpy as np

try:
    import emcee

    emcee_avail = True
except ImportError:
    emcee_avail = False


def logprior_uniform(x: np.ndarray, bounds: np.ndarray) -> float:
    """Logarithm of the non-normalized joint uniform prior.

    This is the default prior distribution.

    Args:
        x: (ndim,) Parameter values to evaluate.
        bounds: (ndim, 2,) An array containing the boundaries of the uniform prior. The
            first column of the array contains the lower bounds and the second column
            contains the upper bounds.

    Returns:
        Logarithm of the non-normalized joint uniform prior evaluated at parameter ``x``.
    """
    l_bounds, u_bounds = bounds.T
    if all(np.less(x, u_bounds)) and all(np.greater(x, l_bounds)):
        ret = 0.0
    else:
        ret = -np.inf

    return ret

class MCMC(emcee.EnsembleSampler):
    """Sampler class for affine invariant MCMC via emcee Python package.

    Args:
        loss: Loss function. Loss(x) = RMSE where [x] = np.ndarray and [RMSE] = float
        nwalkers: Number of walkers to simulate. The minimum number of walkers is
            twice the number of parameters. It defaults to this minimum value.
        T: Sampling temperatures, used to inflate the likelihood function in the MCMC
            sampling. It defaults to the natural temperature :math:`T_0` [Frederiksen2004]_.
        logprior_fn: A function that evaluate logarithm of the prior distribution.
            The prior doesn't need to be normalized. It defaults to a uniform prior
            over a finite range.
        logprior_args: Additional positional arguments of the ``logprior_fn``. If the
            default ``logprior_fn`` is used, then the boundaries of the uniform prior
            can be specified here.
        **kwargs: Additional keyword arguments for ``emcee.EnsembleSampler``.

    Attributes:
        loss: Loss function instance from :class:`~kliff.loss.Loss`
        T: Values of the sampling temperature.
        sampler: Sampler instance from ``emcee.EnsembleSampler``

    Notes:
        As a convention, KLIFF inflates the likelihood by some sampling temperature,
        i.e., :math:`L(\\theta) \propto \exp(-C(\\theta) / T)`. As a default, the
        sampling temperature is set to the natural temperature. To use the untempered
        likelihood (:math:`T=1`), user should specify the argument ``T=1``.


    References:
        .. [Frederiksen2004] S. L. Frederiksen, K. W. Jacobsen, K. S. Brown, and J. P.
            Sethna, “Bayesian Ensemble Approach to Error Estimation of Interatomic
            Potentials,” Phys. Rev. Lett., vol. 93, no. 16, p. 165501, Oct. 2004,
            doi: 10.1103/PhysRevLett.93.165501.
    """

    def __init__(
        self,
        loss,
        ndim,
        bounds,
        nwalkers: Optional[int] = None,
        T: Optional[float] = None,
        logprior_fn: Optional[Callable] = None,
        logprior_args: Optional[tuple] = None,
        **kwargs,
    ):
        self.loss = loss

        # Dimensionality
        nwalkers = 2 * ndim if nwalkers is None else nwalkers
        self.bounds = bounds

        # Probability
        global loglikelihood_fn

        def loglikelihood_fn(x):
            return _get_loglikelihood(x, loss)

        if logprior_fn is None:
            logprior_fn = logprior_uniform
            if logprior_args is None:
                logprior_args = (self.bounds,)

        # Probability
        if T is None:
            self.T = 1
        else:
            self.T = T
        logl_fn = self._loglikelihood_wrapper
        logp_fn = self._logprior_wrapper(logprior_fn, *logprior_args)

        global logprobability_fn

        def logprobability_fn(x):
            return logl_fn(x) + logp_fn(x)

        super().__init__(nwalkers, ndim, logprobability_fn, **kwargs)

    def _loglikelihood_wrapper(self, x):
        """A wrapper to the log-likelihood function, so that the only argument is the
        parameter values.
        """
        return _get_loglikelihood(x, self.loss, self.T)

    def _logprior_wrapper(self, logprior_fn, *logprior_args):
        """A wapper to the log-prior function, so that the only argument is the
        parameter values.
        """
        if logprior_fn is None:
            if logprior_args is None:
                logprior_args = (self.bounds,)
            logprior_fn = logprior_uniform

        def logp(x):
            return logprior_fn(x, *logprior_args)

        return logp


def _get_loglikelihood(x: np.ndarray, loss_fxn, T: Optional[float] = 1.0):
    """Compute the log-likelihood from the cost function. It has an option to temper the
    cost by specifying ``T``.
    """
    cost = loss_fxn(x)
    logl = -cost / T
    return logl

def _get_loglikelihood_FineGrain(x,loss_fxn,C0):
    """assumes loss_fxn = ||y_ab - y_fit|| """
    cost = loss_fxn(x,form="vectorized")
    cost_sq = cost**2
    C0_sq = C0**2
    logl = -np.sum(cost_sq/C0_sq + np.log(1/np.sqrt(2*np.pi*C0_sq)))
    return logl

class MCMCError(Exception):
    def __init__(self, msg):
        super(MCMCError, self).__init__(msg)
        self.msg = msg