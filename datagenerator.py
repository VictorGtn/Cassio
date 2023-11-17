#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fonctions de génération de données en 2D (nuages de points sympathiques).
Certaines inspirées de Numerical Tours/G. Peyré.
Fonctions disponibles:

Created 2021/04/19
@author: castella
"""
import numpy as np
# from typing import NamedTuple


# class Dataset(NamedTuple):
#     data: np.array
#     bounds: np.array
#     classif: np.array


NVAR = 2                        # generation de données en 2D seulement


def genlatentbin(nsamples, eta, exact=False):
    """
    Generate latent (hidden) binary 1D process.

    Each entry is Bernouilli with P(False) = eta, P(True)=1-eta
    TODO: extend e.g. Markov dependence

    Parameters
    ----------
    nsamples : int
    eta : float 0<eta<1
    exact : (optional). It False (default), each sample is randomly driven.
            If exact=True, exactly int(eta*nsamples) are set to False.

    Returns
    -------
    r : ndarray, shape (nsamples, ) with bool entries
    """
    if exact is False:
        r = np.random.rand(nsamples) > eta
    elif exact is True:
        nsamples0 = int(nsamples*eta)
        nsamples1 = nsamples-nsamples0
        r = np.r_[np.repeat(False, nsamples0), np.repeat(True, nsamples1)]
        np.random.default_rng().shuffle(r)  # in-place shuffling
    return r


def mog(nsamples, mu=0.5, sigma=None):
    """
    Generates samples in R from a specific mixture of Gaussians in ref. [1]

    1/2*N(mu,sigma^2) + 1/2*N(-mu,sigma^2)]    with:  0<mu<1
    Default value for sigma is sigma = sqrt(1 - mu^2) to ensure unit variance

    Parameters
    ----------
    nsamples : int
    mu : float
         0<mu<1. Default: mu = 0.5 (value used in [1])
    sigma : float
            Default: sigma = None and is calculated to ensure unit variance

    Returns
    -------
    a : ndarray, shape (nsamples, )

    References
    ----------
    [1] Castella, M., Rafi, S., Comon, P., & Pieczynski, W. (2013). Separation
        of instantaneous mixtures of a particular set of dependent sources
        using classical ICA methods. EURASIP J. Adv. Signal Process., (62), .
    """
    if sigma is None:
        sigma = np.sqrt(1-mu**2)
    a = mu*np.sign(np.random.randn(nsamples)) + sigma*np.random.randn(nsamples)
    return a


def dependsourex1(nsamples, mu=0.5):
    """
    Generates dependent sources according to example 1 in ref. [1]

    Sources are given by s1 = a  s2=epsilon*a, where a obtained by mog.

    Parameters
    ----------
    nsamples : int
    mu : parameter value for mog (mixture of Gaussian)

    Returns
    -------
    s : ndarray, shape (2, nsamples)

    References
    ----------
    [1] Castella, M., Rafi, S., Comon, P., & Pieczynski, W. (2013). Separation
        of instantaneous mixtures of a particular set of dependent sources
        using classical ICA methods. EURASIP J. Adv. Signal Process., (62), .

    TODO:
    -----
    instead of mu, entry of function should be function generating samples
    """
    a = mog(nsamples, mu)
    epsilon = np.sign(np.random.randn(nsamples))
    s = np.c_[a, epsilon*a].T
    return s


def dependsourex2(nsamples, lambd):
    """
    Generates dependent sources according to example 2 in ref. [1]

    Parameters
    ----------
    nsamples : int

    Returns
    -------
    s : ndarray, shape (2, nsamples)

    References
    ----------
    [1] Castella, M., Rafi, S., Comon, P., & Pieczynski, W. (2013). Separation
        of instantaneous mixtures of a particular set of dependent sources
        using classical ICA methods. EURASIP J. Adv. Signal Process., (62), .
    """
    sigma = np.sqrt(2*(1-1/lambd**2))
    u1 = sigma*np.random.randn(nsamples)
    u2 = np.random.default_rng().laplace(0, 1/lambd, nsamples)
    U = np.c_[u1, u2].T
    rot = 1/np.sqrt(2)*np.array([[1, -1],
                                 [1, 1]])
    # ### ci-dessous, cas exact article [1]. Rq: det(rot)<0. Ne change rien ici
    # rot = 1/np.sqrt(2)*np.array([[1, 1],
    #                              [1, -1]])
    s = rot@U
    return s

