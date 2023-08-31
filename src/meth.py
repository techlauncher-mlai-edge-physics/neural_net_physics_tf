"""
Implementation of update steps.
"""
import warnings
import torch
from linear_operator.operators import DiagLinearOperator, LowRankRootLinearOperator
from linear_operator.utils.errors import NotPSDError

from math import sqrt, log, exp
from collections import namedtuple
from time import perf_counter
from .math_helpers import gaussian_log_prob,  NumericalError
from src.iterators import multi_sim_generator
from src.formatting import pre_curry


MethUpdate = namedtuple(
    'MethUpdate',
    ['U', 'sig2_hat', 'tau2_hat', 'timing'])
# LangevinMove = namedtuple('LangevinMove', ['delta_z', ])


def mean_dev(ens):
    """
    man and deviation matrix of an ensemble
    """
    m = ens.mean(dim=0)
    # transpose matrices so that obs are col vectors and notation matches the paper
    dev = (ens - m).T / sqrt(ens.shape[0]-1)
    return m, dev


def ens_log_prob(
            obs,
            ens,
            tau2,
        ):
    """
    A Gaussian likelihood.
    Potentially expensive.
    However, likelihood evaluation is only used to assess model quality; it does not count against the method time cost with fixed hyperparameters.
    """
    m, dev = mean_dev(ens)
    return torch.distributions.LowRankMultivariateNormal(
        m, dev, torch.as_tensor(tau2, device=m.device).tile(m.shape[0])
    ).log_prob(obs)


def update_(
        predictor,
        U,
        obs,
        sig2=0.0,      # forward prediction noise in ensemble estimate
        tau2=0.0,      # Inference noise; Does not arise in Matheron update
        kappa2=0.0,    # perturb ensemble inputs with white noise; unused
        n_update_steps=1,
        method='lowrank',  # if 'lowrank' exploit low rank structure of covariance matrix
        verbose=0):
    """
    takes in batched prior vectors, predicts the system forwards using them forward and uses the joint moments to update the batch of priors in-place.
    """
    if verbose >= 1:
        warnings.warn("Verbose output currently not supported as the gain matrix is no longer explicitly computed.", DeprecationWarning)

    pred_time = 0.0
    la_time = 0.0
    if n_update_steps != 1:
        raise ValueError("this is not an iterative method any more.")
    start_time = perf_counter()
    kappa2 = torch.as_tensor(kappa2, device=U.device)
    sig2 = torch.as_tensor(sig2, device=U.device)
    if kappa2 > 0.0:
        U = U + torch.randn_like(U) * torch.sqrt(kappa2)

    pred_start_time = perf_counter()
    Z = predictor(U).detach()
    pred_time += perf_counter() - pred_start_time
    m_z, Z_dev  = mean_dev(Z)
    m_u, U_dev  = mean_dev(U)
    

    pred_error = (Z - obs).T
    la_start = perf_counter()
    # if method=='lowrank':
    #     K_zz = LowRankRootLinearOperator(
    #         Z_dev).add_diagonal(sig2)
    #     try:
    #         gain = torch.linalg.solve(
    #             K_zz,
    #             U_dev.matmul(Z_dev.T).T).T
    #     except NotPSDError as e:
    #         print("NotPSDError", e)
    #         print("sig2", sig2)
    #         raise NumericalError(f"Failed because {e}")
    #     del(K_zz)
    # elif method=='full':
    #     K_zz = torch.matmul(Z_dev, Z_dev.T)
    #     d_ = torch.diagonal(K_zz)
    #     d_ += sig2
    #     gain = torch.linalg.solve(
    #         K_zz,
    #         U_dev.matmul(Z_dev.T).T).T
    #     del(K_zz, d_)
    # else:
    #     raise ValueError(f"method {method} not recognised")
    # la_time += perf_counter() - la_start
    # if verbose>1:
    #     print("gain", gain.shape, gain.mean(), gain.std())
    # step = gain.matmul(pred_error)

    if method=='lowrank':
        K_zz = LowRankRootLinearOperator(
            Z_dev).add_diagonal(sig2)
        try:
            step = torch.linalg.solve(
                K_zz,
                pred_error)
        except NotPSDError as e:
            print("NotPSDError", e)
            print("sig2", sig2)
            raise NumericalError(f"Failed because {e}")
        del(K_zz)
    elif method=='full':
        K_zz = torch.matmul(Z_dev, Z_dev.T)
        d_ = torch.diagonal(K_zz)
        d_ += sig2
        step = torch.linalg.solve(
            K_zz,
            pred_error)
        del(K_zz, d_)
    else:
        raise ValueError(f"method {method} not recognised")
    la_time += perf_counter() - la_start
    step = U_dev @ (Z_dev.T @ step)


    # this looks like it might be a valid tau2 guesstimate 
    tau2_hat = step.var()
    sig2_hat = pred_error.var()
    # re-transpose so batch index is first again
    U = U - step.T

    total_time = perf_counter() - start_time
    inf_time = total_time - pred_time
    return MethUpdate(
        U,
        tau2_hat,
        sig2_hat,
        dict(
            pred_time=pred_time,
            inf_time=inf_time,
            total_time=total_time,
            la_time=la_time,
        )
    )


def sig2tau2_opt_t(
        init_rand_t,
        simulate_t,
        n_batch,
        sig2_0=1e-1,
        tau2_0=1e-5,
        n_step=5,
        n_sim=10,
        verbose=False,
        method='lowrank',
        do_overs=1,
        alpha=0.75,  # smoothing for the smoothed ML estimate
        ):
    """
    Optimise the method hyperparams by smoothed ML guesstimate of the likelihood of data generated freshly from the simulator.

    This is a bit of a hack, but it outperforms fancier methods, either because it is optimal, or because it is too easy to introduce bugs in fancier methods.
    """
    log_sig2 = log(sig2_0)
    log_tau2 = log(tau2_0)
    
    for obs, force_t in multi_sim_generator(
            init_rand_t, simulate_t, n_steps=n_step, n_sim=n_sim):
        particle_t, velocity_t = obs[0]
        _, _, U = init_rand_t(n_batch=n_batch)

        for next_particle_t, next_velocity_t in obs[1:]:
            _model_pred = pre_curry(simulate_t, particle_t, velocity_t)
            # def _model_pred(force_t):
            #     return simulate_t(particle_t, velocity_t, force_t)[0]
            for _ in range(do_overs):
                U, sig2_hat, tau2_hat, *_ = update_(
                    _model_pred,
                    U.detach(),
                    next_particle_t.detach(),
                    sig2=exp(log_sig2),
                    tau2=exp(log_tau2),
                    method=method,)
                if verbose:
                    print(f"sig2_hat {sig2_hat}, tau2_hat {tau2_hat}")
                # update the smoothed ML estimate
                log_sig2 = alpha * log_sig2 + (1-alpha) * log(sig2_hat)
                log_tau2 = alpha * log_tau2 + (1-alpha) * log(tau2_hat)
                if verbose:
                    print(f"sigma2 {exp(log_sig2)}, tau2 {exp(log_tau2)}")
                
            particle_t = next_particle_t
            velocity_t = next_velocity_t

    return (
        exp(log_sig2),
        exp(log_tau2),
    )


# def hyperparam_opt_t(
#         init_rand_t,
#         simulate_t,
#         n_batch,
#         sig2_0=1e-1,
#         tau2_0=1e-5,
#         kappa2_0=1e-5,
#         n_step=5,
#         n_sim=10,
#         lr=1e-1,
#         verbose=False,
#         sig2_floor=1e-5,
#         tau2_floor=5e-6,
#         kappa2_floor=1e-7,
#         method='lowrank',
#         do_overs=1,
#         hat_alpha=0.5,):
#     """
#     Optimise the method hyperparams by maximising the likelihood of data generated freshly from the simulator.
#     Currently does not work; check derivation
#     """
#     U = init_rand_t(n_batch=0)[-1]
#     log_sig2 = torch.tensor(
#         log(sig2_0),
#         requires_grad=True,
#         dtype=U.dtype,
#         device=U.device)
#     log_tau2 = torch.tensor(
#         log(tau2_0),
#         requires_grad=True,
#         dtype=U.dtype,
#         device=U.device)
#     log_kappa2 = torch.tensor(
#         log(kappa2_0),
#         requires_grad=True,
#         dtype=U.dtype,
#         device=U.device)
#     del(U)
    
#     # optimizer = torch.optim.Adam(
#     #     [log_sig2, log_tau2, log_kappa2],
#     #     lr=lr,
#     #     betas=(0.9, 0.99)
#     # )
#     optimizer = torch.optim.SGD(
#         [log_sig2, log_tau2, log_kappa2],
#         lr=lr,
#     )

#     for obs, force_t in multi_sim_generator(
#             init_rand_t, simulate_t, n_steps=n_step, n_sim=n_sim):
#         particle_t, velocity_t = obs[0]
#         # U = U_0.clone().detach()
#         _, _, U = init_rand_t(n_batch=n_batch)

#         if verbose:
#             logprob = ens_log_prob(force_t, U, torch.exp(log_tau2)).detach()
#             print(f"prior logprob {logprob}")

#         for next_particle_t, next_velocity_t in obs[1:]:
#             optimizer.zero_grad()
#             _model_pred = pre_curry(simulate_t, particle_t, velocity_t)
#             # def _model_pred(force_t):
#             #     return simulate_t(particle_t, velocity_t, force_t)[0]
#             for _ in range(do_overs):
#                 U_, tau2_hat, sig2_hat, *_ = update_(
#                     _model_pred,
#                     U.detach(),
#                     next_particle_t.detach(),
#                     sig2=torch.exp(log_sig2),
#                     tau2=torch.exp(log_tau2),
#                     kappa2=torch.exp(log_kappa2),
#                     method=method,)
#                 if verbose:
#                     print(f"sig2_hat {sig2_hat}, tau2_hat {tau2_hat}")
#                 try:
#                     neglogprob = -ens_log_prob(
#                         force_t, U_, torch.exp(log_tau2))
#                 except ValueError:
#                     ## numerically unstable. The best we can do is hope this tiny value is OK
#                     warnings.warn(f"Numerically unstable at {log_sig2}, {log_tau2}, {log_kappa2}")
#                     break
#                 neglogprob.backward()
#                 torch.nn.utils.clip_grad_value_([log_sig2, log_tau2, log_kappa2], 10)
#                 if verbose:
#                     print(f"logprob {-neglogprob.detach()}\t grads {log_sig2.grad}, {log_tau2.grad}, {log_kappa2.grad}")
#                 optimizer.step()
#                 # hack: avoid numerical explosions
#                 log_sig2.data.clamp_(min=log(sig2_floor))
#                 log_tau2.data.clamp_(min=log(tau2_floor))
#                 log_kappa2.data.clamp_(min=log(kappa2_floor))
            
#                 if verbose:
#                     print(f"sigma2 {torch.exp(log_sig2).detach()}, tau2 {torch.exp(log_tau2).detach()}, kappa2 {torch.exp(log_kappa2).detach()}")
                
#             U = U_.detach()
#             particle_t = next_particle_t
#             velocity_t = next_velocity_t

#     return (
#         torch.exp(log_sig2).detach(),
#         torch.exp(log_tau2).detach(),
#         torch.exp(log_kappa2).detach(),
#         -neglogprob.detach()
#     )



# def ensemble_langevin_step(
#         z, sig2, epsilon, Z_dev=None, m_z=None, method='full', nystrom_samples=1000):
#     """
#     Naive computation of an ensemble-wise Langevin update for a Z-ensemble by Euler-Maruyama integration.
#     More sophisticated approaches might do a Metropolis adjustment or an implicit step.
#     Cheaper methods might use a low rank approximation e.g. via Nyström.
#
#     Sig2 regularizes the implied variance (smaller sig2=bigger update),

#     epsilon is the step size; the larger the step size, the more the update is biased.
#     """
#     with torch.no_grad():
#         m_z = z.mean(dim=0)
#         if Z_dev is None:
#             # transpose matrices so that obs are col vectors and notation matches the paper
#             Z_dev = (z - m_z).T / sqrt(z.shape[0]-1)
#         if method == 'full':
#           delta_z = torch.linalg.solve(
#               torch.matmul(Z_dev, Z_dev.T)
#               + sig2 * torch.eye(Z_dev.shape[-2]),
#               (z - m_z).T
#           ).T
#         ## Insert Nystrom approx here?
#
#         return LangevinMove(
#             delta_z + sqrt(2*epsilon) * torch.randn_like(z),
#         )

