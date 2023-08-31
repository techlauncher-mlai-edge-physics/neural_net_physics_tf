"""
Implementation of update steps.

This is not consistent about in-place versus allocated updates.
"""

import warnings
import torch
from torch.autograd.functional import jacobian

from math import log, exp
from collections import namedtuple
from time import perf_counter
from .math_helpers import gaussian_log_prob, rsolve
from src.iterators import multi_sim_generator
from src.formatting import pre_curry

EKFUpdate = namedtuple(
    'EKFUpdate',
    ['m_u',  'K_uu',  'timing'])


def update_(
        predictor,
        m_u,
        K_uu,
        obs,
        sig2=0.0,    # diffuse observation likelihood
        tau2=0.0,    # diffuser posterior likelihood
        verbose=0):
    """
    takes in prior moments, constructs predictive moment moments from them, uses the joint moments to estimate posterior moments and return.
    """
    pred_time = 0.0
    la_time = 0.0
    start_time = perf_counter()
    pred_start_time = perf_counter()
    m_z = predictor(m_u)
    J_u = jacobian(
        predictor,
        m_u)
    # we do not want to differentiate through the derivatives
    J_u = J_u.detach_()
    m_z = m_z.detach()
    pred_time += perf_counter() - pred_start_time
    la_start = perf_counter()
    # apparently we need to construct the eye matrix to allow this to be a differentiable op
    K_zz = J_u @ K_uu @ J_u.T + sig2 * torch.eye(
        m_z.shape[0], device=K_uu.device)
    gain = rsolve(K_uu @ J_u.T, K_zz)
    K_uu_hat = K_uu - gain @ J_u @ K_uu
    if tau2 > 0:
        K_uu_hat = K_uu_hat + tau2 * torch.eye(K_uu_hat.shape[0], device=K_uu_hat.device)
    m_u_hat = m_u + gain @ (obs - m_z)
    la_time += perf_counter() - la_start
    if verbose>1:
        print("gain", gain.shape, gain.mean(), gain.std())
    # re-transpose so batch index is first agin

    total_time = perf_counter() - start_time
    inf_time = total_time - pred_time
    return EKFUpdate(
        m_u_hat,
        K_uu_hat,
        dict(
            pred_time=pred_time,
            inf_time=inf_time,
            total_time=total_time,
            la_time=la_time,
        )
    )


def hyperparam_opt(
        init_rand_t, simulate_t,
        m_u_0, K_uu_0,
        sig2_0=1e-1,
        tau2_0=1e-5,
        n_step=5, n_sim=10,
        lr=5e-1, verbose=False,
        sig2_floor=1e-5,
        tau2_floor=5e-6):
    
    log_sig2 = torch.tensor(
        log(sig2_0),
        requires_grad=True,
        dtype=m_u_0.dtype)
    log_tau2 = torch.tensor(
        log(tau2_0),
        requires_grad=True,
        dtype=m_u_0.dtype)
    
    optimizer = torch.optim.Adam(
        [log_sig2, log_tau2],
         lr=lr, betas=(0.5, 0.9))

    for obs, force_t in multi_sim_generator(
            init_rand_t, simulate_t, n_steps=n_step, n_sim=n_sim):
        particle_t, velocity_t = obs[0]
        m_u = m_u_0.clone().detach()
        K_uu = K_uu_0.clone().detach()
        if verbose:
            logprob = gaussian_log_prob(force_t, m_u, K_uu).detach()
            print(f"prior logprob {logprob}")

        for next_particle_t, next_velocity_t in obs[1:]:
            optimizer.zero_grad()
            _model_pred = pre_curry(simulate_t, particle_t, velocity_t)
            
            m_u, K_uu, *_ = update_(
                _model_pred,
                m_u.detach(),
                K_uu.detach(),
                next_particle_t.detach(),
                sig2=torch.exp(log_sig2),
                tau2=torch.exp(log_tau2))
            try:
                neglogprob = -gaussian_log_prob(
                    force_t, m_u, K_uu)
            except ValueError:
                ## numerically unstable. The best we can do is hope this tiny value is OK
                warnings.warn(f"Numerically unstable at {log_sig2}, {log_tau2}")
                break
            neglogprob.backward()
            torch.nn.utils.clip_grad_value_([log_sig2, log_tau2], 10)
            if verbose:
                print(f"logprob {-neglogprob.detach()}\t grads {log_sig2.grad}, {log_tau2.grad}")
            optimizer.step()
            # hack: avoid numerical explosions
            log_sig2.data.clamp_(min=log(sig2_floor))
            log_tau2.data.clamp_(min=log(tau2_floor))

            if verbose:
                print(f"sigma2 {torch.exp(log_sig2).detach()}, tau2 {torch.exp(log_tau2).detach()}")
            particle_t = next_particle_t
            velocity_t = next_velocity_t
    return (
        torch.exp(log_sig2).detach(),
        torch.exp(log_tau2).detach(),
        -neglogprob.detach()                         
    )
