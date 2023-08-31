"""
iterators and generators through sims in various useful ways for NN training.
"""

import sys
from . import formatting


def sim_generator(
        simulate, n_steps, particle, velocity, force, pressure=None,
        stride=1, *args, **kwargs):
    """
    generate a simulation.
    return steps one at a time, including inputs
    """
    yield particle, velocity, pressure
    for i in range(1, (n_steps - 1) * stride + 1):
        particle, velocity, pressure = simulate(
            particle, velocity, force, pressure, *args, **kwargs)
        if i % stride == 0:
            yield particle, velocity, pressure


def last_n(xs, n_context):
    """
    return a new iterable made up of the last few items.
    Could be replaced with `itertools.sliding_window`.
    """
    history = []
    for x in xs:
        history.append(x)
        if len(history) >= n_context:
            yield list(history) #return a copy
            history.pop(0)


def corrupt(it, *args, **kwargs):
    """
    Apply a noisy perturbation to a stream.
    This is for observations only; the process noise is handled elsewhere.
    """
    corrupt = formatting.corruptor(*args, **kwargs)
    return map(corrupt, it)


def multi_sim_context_generator(
        init_rand, simulate, n_steps, n_context=2,
        stride=1,
        *args, max_steps=sys.maxsize, n_batch=1, **kwargs):
    """
    chain lots of sims together and return the last_n sets for packing into the history for e.g. a dataloader
    """
    steps = 0
    while True:
        particle, velocity, force = init_rand(n_batch=n_batch)
        # Why do we pack the force here?
        force_packed = formatting.as_cattable_field(force)
        for sim_i in last_n(
                sim_generator(
                    simulate, n_steps, particle, velocity, force, stride, *args, **kwargs),
                n_context):
            yield sim_i + [force_packed]
            steps += 1
            if steps >= max_steps:
                return


def multi_sim_generator(
        init_rand, simulate,
        n_steps,  # steps per sim
        *args,
        n_sim=1,  # sims
        **kwargs):
    """
    returns many different problems sequentially.
    """
    for sim_i in range(n_sim):
        pressure = None
        particle, velocity, force = init_rand(n_batch=0)
        obs = [(particle, velocity),]
        for step_i in range(n_steps):
            particle, velocity, pressure = simulate(
                particle, velocity, force, pressure=pressure, *args, **kwargs)
            particle, velocity, force
            obs.append((particle, velocity,))
        
        yield obs, force
