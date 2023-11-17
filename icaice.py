#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Combined ICA/ICE procedure as in:

Castella, M., Rafi, S., Comon, P., & Pieczynski, W., Separation of
instantaneous mixtures of a particular set of dependent sources using
classical ICA methods, EURASIP J. Adv. Signal Process., (62), (2013).
"""
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import datagenerator as mdg
import bssica


def icaice(x, niter, estim_eta=True, pihat_ini=np.array([0.5, 0.5]), nice=1,
           lambda_ICE=5):
    """Combined ICA/ICE procedure as described in [1]

    Parameters
    ----------
    x : array_like, shape 2 x Nsamples
        mixed sources to be separated
    niter : int (default = 30)
        number of iterations for ICE
    estim_eta : bool, estimate probability for hidden state process.
        (default = True). If False, pihat_ini will be used.
    pihat_ini : 1D array
        initial probability values for hidden state process P(r) (default =
        uniform distribution)
    nice : int
        number of drawings for ICE stochastic approximation (default = 1)
    lambda_ICE : float (default = 5)
        parameter for dependent par of distribution

    Returns
    -------
    Bhat : array_like 2x2
        estimated separating matrix
    r_ice : array_like with bool 1xlen(x)
        last value of stochastic approximation of hidden process

    References
    ----------
    .. [1] Castella, M., Rafi, S., Comon, P., & Pieczynski, W., Separation of
           instantaneous mixtures of a particular set of dependent sources
           using classical ICA methods, EURASIP J. Adv. Signal Process., (62),
           (2013).
    """
    # initialization of the parameters
    _, T = x.shape
    pihat = pihat_ini
    Ahat = bssica.comica(x)
    Bhat = np.linalg.pinv(Ahat)  # np.linalg.pinv dans matlab -> inv?

    Pxr = np.empty((2, T))
    for iter in range(niter):       # ICE iterations
        # --- calculate prob. conditionnally to r: Pxr(i,t) = Prob(x(t)/r(t)=i)
        y = Bhat@x

        # % conditionnally to r=0 (independent sources)
        # %%-- Prob(x(t)/r(t)=0) is assumed gaussian
        sigmatmp0 = 1
        Pxr[0, :] = 1/(2*np.pi*sigmatmp0**2)*np.exp(
            -(y**2).sum(axis=0)/(2*sigmatmp0**2))

        # % conditionnally to r=1 (dependent sources)
        # %%-- along the 2 bisectors, Gaussian x Laplace
        rot = np.array([[1, 1], [1, -1]])/np.sqrt(2)
        z = rot@y
        lambdatmp1 = lambda_ICE
        sigmatmp1 = np.sqrt(2*(1-1/lambdatmp1**2))

        mulfac = 1/2*1/(sigmatmp1*np.sqrt(2*np.pi))*lambdatmp1/2
        Pxr[1, :] = mulfac*(np.exp(-z[0, :]**2/(2*sigmatmp1**2))
                            * np.exp(-lambdatmp1*abs(z[1, :]))
                            + np.exp(-z[1, :]**2/(2*sigmatmp1**2))
                            * np.exp(-lambdatmp1*abs(z[0, :])))

        Pconjointexr = np.diag(pihat)@Pxr
        # attention: si pihat était colonne, np.diag extrait diagonale
        Pposteriorir = Pconjointexr/Pconjointexr.sum(axis=0)  # broadcasting

        # --- re-estimation of the parameters ---
        if estim_eta is True:
            pihat = Pposteriorir.sum(axis=1)/T
            # pihat = 1/T*sum(Pposteriorir,2);

            # --- stochastic approximation of ICE ---
            Bhattmp = list()
            for iter in range(nice):
                r_ice = np.random.rand(T) > Pposteriorir[0, :]
                Ahat = bssica.comica(x[:, ~r_ice])
                Bhattmp.append(np.linalg.inv(Ahat))
                Bhat = reduce(np.add, Bhattmp)/nice  #
    return Bhat, r_ice


if __name__ == '__main__':
    # generate data
    T = 2000                        # number of samples
    eta = 0.5        # eta = Prob(r=0) = Prob(s1 and s2 independent)
    # generating the auxiliary process r (in this program, r is iid)
    r = mdg.genlatentbin(T, eta, exact=False)
    Nsamples0, Nsamples1 = (~r).sum(), r.sum()  # keep parenthesis!
    s0 = bssica.generatesources(2, Nsamples0, 'rand')
    s1 = mdg.dependsourex1(Nsamples1)
    s = np.empty((2, T))
    s[:, ~r] = s0
    s[:, r] = s1
    # Mixing
    A = np.random.randn(2, 2)
    x = A@s

    # separation
    Bhat_ice, r_ice = icaice(x, 30, estim_eta=True,
                             pihat_ini=np.array([0.5, 0.5]), nice=1)
    # %-- final estimates --
    G_ice = Bhat_ice@A
    y = Bhat_ice@x
    mse_ice, _ = bssica.elimamb(y, s)
    segrate_ice = (r_ice == r).sum()/T
    # contestable, pourquoi pas pour N_ICE = 1

    # %% -- display and print results --
    plt.ion()
    fig, ax = plt.subplots(num=1)
    ax.plot(s[0, ~r_ice], s[1, ~r_ice], 'b.')
    ax.plot(s[0, r_ice], s[1, r_ice], 'g.')
    # plt.show()
    print("--- Combined ICA/ICE results -----------")
    print(f"Segmentation rate: {segrate_ice}")
    print("Glogal mixing-separating matrix G "
          "(should be identity up to ambiguities)")
    print(G_ice)
    print("Separation criterion calculated on G:  "
          f"{1-(G_ice**2).max(axis=1)/(G_ice**2).sum(axis=1)}")
    print(f"MSE on each source {mse_ice}")

    print("--- Compare when ignoring latent model and dependence ---")
    Bhat = np.linalg.pinv(bssica.comica(x))
    Ghat = Bhat@A
    y = Bhat@x
    mse, _ = bssica.elimamb(y, s)
    print("Glogal mixing-separating matrix G "
          "(should be identity up to ambiguities)")
    print(Ghat)
    print("Separation criterion calculated on G:  "
          f"{1-(Ghat**2).max(axis=1)/(Ghat**2).sum(axis=1)}")
    print(f"MSE on each source {mse}")

    print("--- Compare when supervised (known latent process r) ---")
    Bhat_know = np.linalg.pinv(bssica.comica(x[:, ~r]))
    Ghat_know = Bhat_know@A
    y = Bhat_know@x
    mse_know, _ = bssica.elimamb(y, s)
    print("Glogal mixing-separating matrix G "
          "(should be identity up to ambiguities)")
    print(Ghat_know)
    print("Separation criterion calculated on G:  "
          f"{1-(Ghat_know**2).max(axis=1)/(Ghat_know**2).sum(axis=1)}")
    print(f"MSE on each source {mse_know}")
