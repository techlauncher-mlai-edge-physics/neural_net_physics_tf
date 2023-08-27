from math import sqrt, ceil, floor
from phi import math
import itertools

from . import meth, formatting, physical_models, ekf
from .math_helpers import  NumericalError

def ncycles(iterable, n):
    "Returns the sequence elements n times"
    return itertools.chain.from_iterable(itertools.repeat(tuple(iterable), n))


def setup_problem(
        grid_size=(32,32),
        v_noise_power: float =100,
        TORCH_DEVICE = 'cpu',  # not needed?
        PHI_DEVICE ="CPU",
        seed: int = 32,
        n_skip_steps: int =4,
        n_obs: int=5):
    grid_size = (32,32)
    grid_size_x, grid_size_y = grid_size
    math.seed(seed)

    init_rand, simulate = physical_models.ns_sim(
        phi_device=PHI_DEVICE,
        grid_size=grid_size,
        jit=False,
        incomp=False,
        v_noise_power=v_noise_power,
        n_skip_steps=n_skip_steps)

    #truth
    particle, velocity, force = init_rand(n_batch=0)
    obs = [[particle, velocity]]

    for i in range(1, n_obs):
        particle, velocity, _ = simulate(
            particle, velocity, force, n_skip_steps=n_skip_steps)
        obs.append([particle, velocity])

    f_wrapper = formatting.TorchVectorWrapper(
        force,
        x=grid_size_x,
        y=grid_size_y,
        vector=2,
    )
    p_wrapper = formatting.TorchVectorWrapper(
        particle,
        x=grid_size_x,
        y=grid_size_y
    )
    return init_rand, simulate, force, f_wrapper, p_wrapper, obs
    

def exp_infer(
        method='metho',
        *args, **kwargs):
    """
    wrap an experiment up nicely for ease of use in job executors
    """
    if method=='metho':
        return exp_infer_metho(*args, **kwargs)
    elif method=='ekf':
        return exp_infer_ekf(*args, **kwargs)
    else:
        raise ValueError(f"What is this method, `{method}`?")


def exp_infer_meth(
        grid_size=(32,32),
        v_noise_power=100.0,
        TORCH_DEVICE = 'cpu',
        PHI_DEVICE ="CPU",
        seed = 32,
        n_skip_steps=4,
        n_obs=5,
        n_batch=1000,
        verbose=0,
        sig2=0.001,
        tau2=0.001,
        **method_kwargs):
    """
    wrap an experiment up nicely for ease of use in job executors
    """
    init_rand, simulate, force, f_wrapper, p_wrapper, obs = setup_problem(
        grid_size=grid_size,
        v_noise_power=v_noise_power,
        TORCH_DEVICE=TORCH_DEVICE,
        PHI_DEVICE =PHI_DEVICE,
        seed =seed,
        n_skip_steps=n_skip_steps,
        n_obs=n_obs
    )
    _, _, force_guess = init_rand(n_batch=n_batch)
    force_guess_t = f_wrapper.unwrap(force_guess)
    # guesses = [torch.clone(force_guess_t)]
    particle, velocity = obs[0]
    force_t = f_wrapper.unwrap(force)
    particle, velocity = obs[0]
    times = []
    try:
        for i, (next_particle, next_velocity) in enumerate(obs[1:]):
            def forward(force_guess_t):
                """takes in batched input vectors, interprets them as fields, runs sim using the field, returns new batch of vectors representing the forward prediction of interest.
                """
                #wrap for PhiFlow
                force_guess = f_wrapper.wrap(force_guess_t)
                #predict
                particle_pred, _, _ = simulate(
                    particle, velocity, force_guess, n_skip_steps=n_skip_steps)
                #unwrap for pytorch
                return p_wrapper.unwrap(particle_pred)
            force_guess_t, sig2_hat, tau2_hat, *_, time = meth.update_(
                forward,
                force_guess_t,
                p_wrapper.unwrap(next_particle),
                verbose=verbose,
                sig2=sig2,
                tau2=tau2,
                **method_kwargs)
            times.append(time)
            # print("logprob", logprob)
            # guesses.append(torch.clone(force_guess_t))
            particle, velocity = next_particle, next_velocity
    except NumericalError as e:
        return None
    m, _ = meth.mean_dev(force_guess_t)
    l2_loss = sqrt(float(((force_t-m)**2).mean()))
    try:
        logprob = float(
            meth.ens_log_prob(force_t, force_guess_t, tau2=tau2))
    except ValueError as e:  # covariance matrix not PD
        print(e)
        logprob = float('-inf')
    try:
        logprob_hat = float(
            meth.ens_log_prob(force_t, force_guess_t, tau2=tau2_hat))
    except ValueError as e:  # covariance matrix not PD
        print(e)
        logprob_hat = float('-inf')
    return dict(
        logprob=logprob,
        logprob_hat=logprob_hat,
        times=times,
        l2_loss=l2_loss, 
        sig2=sig2,
        tau2=tau2,
        tau2_hat=float(tau2_hat),
        n_batch=n_batch,
        **method_kwargs)


def exp_infer_ekf(
        grid_size=(32,32),
        v_noise_power=100.0,
        TORCH_DEVICE = 'cpu',
        PHI_DEVICE ="CPU",
        seed = 32,
        n_skip_steps=4,
        n_obs=5,
        n_batch=1000,
        verbose=0,
        sig2=0.001,
        tau2=0.001,
        **method_kwargs):
    """
    wrap an experiment up nicely for ease of use in job executors
    """
    init_rand, simulate, force, f_wrapper, p_wrapper, obs = setup_problem(
        grid_size=grid_size,
        v_noise_power=v_noise_power,
        TORCH_DEVICE=TORCH_DEVICE,
        PHI_DEVICE =PHI_DEVICE,
        seed =seed,
        n_skip_steps=n_skip_steps,
        n_obs=n_obs
    )
    _, _, force_guess = init_rand(n_batch=n_batch)
    force_guess_t = f_wrapper.unwrap(force_guess)
    m_u, K_uu = torch.mean(force_guess_t), torch.cov(force_guess_t)
    particle, velocity = obs[0]
    force_t = f_wrapper.unwrap(force)
    particle, velocity = obs[0]
    times = []
    try:
        for i, (next_particle, next_velocity) in enumerate(obs[1:]):
            def forward(m_u, K_uu):
                """takes in batched input vectors, interprets them as fields, runs sim using the field, returns new batch of vectors representing the forward prediction of interest.
                """
                #wrap for PhiFlow
                force_guess = f_wrapper.wrap(force_guess_t)
                #predict
                particle_pred, _, _ = simulate(
                    particle, velocity, force_guess, n_skip_steps=n_skip_steps)
                #unwrap for pytorch
                return p_wrapper.unwrap(particle_pred)
            m_u_hat, K_uu_hat, *_, time = ekf.update_(
                forward,
                force_guess_t,
                p_wrapper.unwrap(next_particle),
                verbose=verbose,
                sig2=sig2,
                tau2=tau2,
                **method_kwargs)
            times.append(time)
            # print("logprob", logprob)
            # guesses.append(torch.clone(force_guess_t))
            particle, velocity = next_particle, next_velocity
    except NumericalError as e:
        return None
    m, _ = meth.mean_dev(force_guess_t)
    l2_loss = sqrt(float(((force_t-m)**2).mean()))
    try:
        logprob = float(
            meth.ens_log_prob(force_t, force_guess_t, tau2=tau2))
    except ValueError as e:  # covariance matrix not PD
        print(e)
        logprob = float('-inf')
    try:
        logprob_hat = float(
            meth.ens_log_prob(force_t, force_guess_t, tau2=tau2_hat))
    except ValueError as e:  # covariance matrix not PD
        print(e)
        logprob_hat = float('-inf')
    return dict(
        logprob=logprob,
        logprob_hat=logprob_hat,
        times=times,
        l2_loss=l2_loss, 
        sig2=sig2,
        tau2=tau2,
        tau2_hat=float(tau2_hat),
        n_batch=n_batch,
        **method_kwargs)

def multi_exp(
        fn,
        seed=89,
        repeat=10,
        *args, **kwargs):
    """a batch of lots of experiments run in a single chunk to be kind to the SLURM scheduler.
    Ignore None results, which are returned when an experiment fails. (Careful that these are not informative)"""
    exps = []
    for i in range(repeat):
        res = fn(*args, seed=seed+i, **kwargs)
        if res is not None:
            exps.append(res)
    return exps
