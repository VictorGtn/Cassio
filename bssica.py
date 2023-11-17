#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module with diverse functions for blind source separation (BSS) and
independent component analysis (ICA)
"""

import numpy as np


def generatesources(Nsources, Nsamples, choice):
    """ Generates sources with independent components (for ICA/BSS)

    Returned sources are normalized.

    Parameters
    ----------
    Nsources : int, number of sources
    Nsamples : int, number of samples
    choice : selected type of sources: 'bpsk', 'rand'

    Returns
    -------
    sources : ndarray Nsour x Nsamples
              realization of the sources
    """
    if choice == 'bpsk':
        sources = np.sign(np.random.randn(Nsources, Nsamples))
    elif choice == 'rand':
        sources = np.random.rand(Nsources, Nsamples) - 0.5

    # renormalize (unit variance)
    if Nsamples == 0:
        sources = np.empty((Nsources, 0))
    else:
        renormfactors = 1/np.sqrt((sources**2).sum(axis=1)/Nsamples)
        sources = sources*np.repeat(renormfactors[:, np.newaxis], Nsamples,
                                    axis=1)
    return sources


def tfuni4(e):
    """Replicates tfuni4.m : from CoM2 algorithm for ICA by P. Comon

    Orthogonal real direct transform for separating 2 sources in presence of
    noise. Sources are assumed zero mean.

    Parameters
    ----------
    e : (2,T)-array

    Returns
    -------
    S : (2,T)-array shape
    A : (2,2)-array shape

    """
    T = max(e.shape)
    # %%%%% moments d'ordre 2
    g11 = sum(e[0, :]*e[0, :])/T        # cv vers 1
    g22 = sum(e[1, :]*e[1, :])/T        # cv vers 1
    g12 = sum(e[0, :]*e[1, :])/T        # cv vers 0
    # %%%%% moments d'ordre 4
    e2 = e**2
    g1111 = sum(e2[0, :]*e2[0, :])/T
    g1112 = sum(e2[0, :]*e[0, :]*e[1, :])/T
    g1122 = sum(e2[0, :]*e2[1, :])/T
    g1222 = sum(e2[1, :]*e[1, :]*e[0, :])/T
    g2222 = sum(e2[1, :]*e2[1, :])/T
    # %%%%% cumulants croises d'ordre 4
    q1111 = g1111-3*g11*g11
    q1112 = g1112-3*g11*g12
    q1122 = g1122-g11*g22-2*g12*g12
    q1222 = g1222-3*g22*g12
    q2222 = g2222-3*g22*g22
    # %%%%% racine de Pw(x): si t est la tangente de l'angle, x=t-1/t.
    u = q1111+q2222-6*q1122
    v = q1222-q1112
    z = q1111*q1111+q2222*q2222

    c4 = q1111*q1112-q2222*q1222
    c3 = z-4*(q1112*q1112+q1222*q1222)-3*q1122*(q1111+q2222)
    c2 = 3*v*u
    c1 = 3*z-2*q1111*q2222-32*q1112*q1222-36*q1122*q1122
    c0 = -4*(u*v+4*c4)

    Pw = np.array([c4, c3, c2, c1, c0])
    R = np.roots(Pw)
    float_epsilon = np.finfo(float).eps
    xx = R[abs(R.imag) < float_epsilon].real
    # %%%%% maximum du contraste en x
    a0 = q1111
    a1 = 4*q1112
    a2 = 6*q1122
    a3 = 4*q1222
    a4 = q2222
    b4 = a0*a0+a4*a4
    b3 = 2*(a3*a4-a0*a1)
    b2 = 4*a0*a0+4*a4*a4+a1*a1+a3*a3+2*a0*a2+2*a2*a4
    b1 = 2*(-3*a0*a1+3*a3*a4+a1*a4+a2*a3-a0*a3-a1*a2)
    b0 = 2*(a0*a0+a1*a1+a2*a2+a3*a3+a4*a4+2*a0*a2+2*a0*a4+2*a1*a3+2*a2*a4)

    Pk = [b4, b3, b2, b1, b0]   # numerateur du contraste
    Wk = np.polyval(Pk, xx)
    Vk = np.polyval([1, 0, 8, 0, 16], xx)
    Wk = Wk/Vk
    Xsol = xx[Wk == max(Wk)]      # [Wmax,j]=max(Wk); Xsol=xx(j);
    # %%%%% maximum du contraste en theta
    t = np.roots(np.array([1, -Xsol[0], -1]))
    mask = np.all([-1 < t, t <= 1], axis=0)
    t = t[mask]
    # %%%%% test et conditionnement
    if abs(t) < 1/T:
        A = np.eye(2)
        # fprintf('pas de rotation plane pour cette paire\n');
    else:
        A = np.empty([2, 2])
        A[0, 0] = 1/np.sqrt(1+t*t)
        A[1, 1] = A[0, 0]
        A[0, 1] = t*A[0, 0]
        A[1, 0] = -A[0, 1]
    # %%%%% filtrage de la sortie
    S = A@e

    return S, A
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def elimamb(x: np.array, y: np.array):
    """ Eliminate permutation/sign ambiguities in ICA and compte MSE

    Parameters
    ----------
    x : array Nx x Nsamples
    y : array Ny x Nsamples with Ny = Nx

    Returns
    -------
    mse : array Nx
    P : array Nx x Nx
    """
    assert (x.shape == y.shape)
    Nsources = x.shape[0]
    Nsamples = x.shape[1]
    r = x@y.T
    P = np.zeros(r.shape)
    for _ in range(Nsources):
        indmax = np.unravel_index(abs(r).argmax(), r.shape)
        P[indmax] = np.sign(r[indmax])
        r[indmax[0], :] = 0
        r[:, indmax[1]] = 0
    newx = x
    newy = P@y
    mse = ((newx-newy)**2).sum(axis=1)/Nsamples

    return mse, P


def comica(Y: np.array):
    """ ICA algorithm Com2 by Pierre Comon (Adapted, original code in Matlab).

    REFERENCE: P.Comon, "Independent Component Analysis, a new concept?",
    Signal Processing, Elsevier, vol.36, no 3, April 1994, 287-314.

    Parameters
    ----------
    Y : array Ny x Nsamples (Ny << Nsamples)

    Returns
    -------
    F : array
    """
    if Y.shape[0] > Y.shape[1]:
        Y = Y.transpose()
    N, T = Y.shape              # Y est maintenant NxT avec N<T.
    # %%%% STEPS 1 & 2: whitening and projection (PCA)
    U, s, V = np.linalg.svd(Y.T, full_matrices=False)
    # ATTENTION: difference svd wrt Matlab: s vector, V<- Vh
    tol = 0                     # original prog: tol=max(size(S))*norm(S)*eps;
    mask = s > tol
    U = U[:, mask]
    V = (V.T)[:, mask]
    S = np.diag(s[mask])
    r = U.shape[1]
    Z = U.T*np.sqrt(T)
    L = V@S.T/np.sqrt(T)
    F = L
    # %%%% STEPS 3 & 4 & 5: Unitary transform
    S = Z
    if N == 2:
        K = 1
    else:
        K = int(1 + np.round(np.sqrt(N)))  # max number of sweeps
    Rot = np.eye(r)
    for k in range(K):
        Q = np.eye(r)
        for i in range(r-1):
            for j in range(i+1, r):
                S1ij = np.vstack((S[i, :], S[j, :]))
                Sij, qij = tfuni4(S1ij)  # %%%%%% processing a pair
                S[i, :] = Sij[0, :]
                S[j, :] = Sij[1, :]
                Qij = np.eye(r)
                Qij[i, i] = qij[0, 0]
                Qij[i, j] = qij[0, 1]
                Qij[j, i] = qij[1, 0]
                Qij[j, j] = qij[1, 1]
                Q = Qij@Q
        Rot = Rot@Q.T
    F = F@Rot
    return F
