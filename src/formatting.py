"""
Utilities to convert between different representations of phiflow Fields, phiflow Tensors, pytorch tensors and numpy ndarrays.

This has sprouted many hairs and could be pruned/refactoed
"""

from itertools import chain

from phi.flow import *
from phi.field import Field
from phi.math import Shape, extrapolation
from einops import rearrange
from collections import namedtuple

import numpy as np

def channel_last_order_for(shape: Shape):
    '''
    Return a tuple of string, represents the order of dimensions
    e.g. ('batch','x','y','vector')
    If the current Shape does not have channel dims, fill in "vector" as 1.
    '''
    batchNames = shape.batch.names if (shape.batch_rank > 0) else ('batch',)
    channelNames = shape.channel.names if (shape.channel_rank > 0) else ('vector',)
    return batchNames + shape.spatial.names + channelNames


def to_ndarray(field: Field) -> np.ndarray:
    '''
    Turn the current Field into ndarray, with shape (batch, x1, ..., xd, v)
    '''
    centered = as_center_grid_field(field)
    order = channel_last_order_for(centered.shape)
    ndarray = centered.values.numpy(order=order)
    return ndarray


def _to_native(t, channels_last=False, force_vector=True):
    """
    copied from `native_call`.

    vector of input tensors are converted to centered native tensors depending on `channels_last`.

    Args:
        t: Uniform `Tensor` arguments
        channels_last: (Optional) Whether to put channels as the last dimension of the native representation.
    Returns:
        `Tensor` with batch and spatial dimensions of `t` and single channel dimension `vector` if `force_vector`.
    """
    t = as_tensor(t)
    if force_vector:
        t = as_vector(t)
    batch = t.shape.batch
    spatial = t.shape.spatial
    groups = (
        (*batch, *spatial.names, *t.shape.channel)
        if channels_last
        else (*batch, *t.shape.channel, *spatial.names)
    )
    return math.reshaped_native(t, groups)


def to_native(*ts, **kwargs):
    """
    vector of input tensors are converted to centered native tensors depending on `channels_last`.

    Args:
        *ts: Uniform `Tensor` arguments
        channels_last: (Optional) Whether to put channels as the last dimension of the native representation.
    """
    return [_to_native(t, **kwargs) for t in ts]


def to_natives(ts, **kwargs):
    """
    Same as `to_native`, but takes a single array argument because that is easier for `map`

    Args:
        ts: Uniform `Tensor` arguments
        channels_last: (Optional) Whether to put channels as the last dimension of the native representation.
    """
    return [_to_native(t, **kwargs) for t in ts]


def to_native_chan_last(*t, **kwargs):
    """
    helper function because it's tedious to pass arguments into iterators.
    calls `to_native` with `channels_last=True`.
    """
    return to_native(*t, channels_last=True, **kwargs)


def to_natives_chan_last(t, **kwargs):
    """
    Same as `to_native`, but takes a single array argument because that is easier for `map`

    helper function because it's tedious to pass arguments into iterators.
    calls `to_natives` with `channels_last=True`.
    """
    return to_natives(t, channels_last=True, **kwargs)


def as_field(
        t,
        bounds=Box(
            x=1.0,
            y=1.0
        ),
        extrapolation=extrapolation.ZERO) -> Field:
    '''
    Turn a native tensor into a centred field
    '''
    return CenteredGrid(
        t,
        bounds=bounds,
        extrapolation=extrapolation)

def as_field_like(
        t,
        prototype) -> Field:
    '''
    Turn a native tensor into a centred field
    '''
    return CenteredGrid(
        t,
        bounds=prototype.bounds,
        extrapolation=prototype.extrapolation)


class TorchVectorWrapper:
    """
    Convert a vector field to a torch tensor and back.
    Batch, if it exists, is first dim. Channels, if defined, are last.
    """

    def __init__(self, prototype, **coords):
        self.prototype = prototype
        self.coords = coords

    def non_batch_name_str(self):
        return " ".join((
            self.prototype.shape.spatial.names 
            + self.prototype.shape.channel.names
        ))

    def non_batch_rank(self):
        return  (
            self.prototype.shape.spatial_rank
            + self.prototype.shape.channel_rank
        )
    
    def non_batch_shape(self):
        return (
            spatial(*self.prototype.shape.spatial.names),
            channel(*self.prototype.shape.channel.names)
        )
    
    def wrap(self, tensor: Tensor) -> Field:
        """
        wrap into a phiflow field
        """
        non_batch_name_str = self.non_batch_name_str()
        if tensor.ndim == 1:
            einops_str = f"({non_batch_name_str}) -> {non_batch_name_str}"
            shape = self.non_batch_shape()
        elif tensor.ndim == 2:
            # presumably this is batched
            einops_str = f"b ({non_batch_name_str}) -> b {non_batch_name_str}"
            shape = (batch("batch"), *self.non_batch_shape())
        else:
            raise ValueError(
                f"vectors shape {tensor.shape} does not conform to prototype shape self.prototype.shape"
            )
        tensor = rearrange(
            tensor,
            einops_str,
            **self.coords)
        tensor = math.tensor(tensor, *shape)
        return as_field_like(
            tensor, self.prototype)

    def unwrap(self, field: Field) -> Tensor:
        non_batch_name_str = self.non_batch_name_str()
        if field.shape.batch_rank==0:
            einops_str = f"{non_batch_name_str} -> ({non_batch_name_str})"
        elif field.shape.batch_rank==1:
            einops_str = f"b {non_batch_name_str} -> b ({non_batch_name_str})"
        else:
            raise ValueError(
                "weird batch shape"
            )
        [vector] = to_native_chan_last(
            field, force_vector=False)
        return rearrange(
            vector,
            einops_str)


def as_vector(t: Tensor)-> Tensor:
    """
    guarantees a channel dimension called vector with size at least 1.
    """
    if len(channel(t.shape)) == 0:
        return math.expand(t, channel(vector=1))
    return t


def as_cattable_tensor(t: Tensor or Field) -> Tensor:
    """
    guarantees a tensor with channel dimension called vector with size at least 1.
    """
    t = as_tensor(t).values
    if len(channel(t)) == 0:
        return math.expand(t, channel(vector=1))
    return t


def as_tensor(t: Tensor or Field) -> Tensor:
    """
    guarantees a tensor
    """
    if isinstance(t, Field):
        t = as_center_grid_field(t).values
    return t


def cat_tensors(ts):
    """
    Densely packs a list of tensors into a single Tensor along a channel dimension, creating if needed.
    """
    return math.concat([
        as_vector(t)
        for t in ts
    ], dim=channel('vector'))


def as_center_grid_field(field: Field, extrap=None) -> CenteredGrid:
    '''
    resample the input `Field` and return a corresponding `CenteredGrid`
    '''
    # `hasattr(field, 'at_centers')` is True even when the method does not exist
    if not isinstance(field, CenteredGrid):
        field = field.at_centers()
    if extrap is not None:
        field = field.with_extrapolation(extrap)
    return field


def as_cattable_field(field: Field, extrap=None) -> CenteredGrid:
    '''
    possibly resample the input `Field` and return a corresponding `CenteredGrid` with a channel dimension of at least 1
    '''
    # `hasattr(field, 'at_centers')` is True even when the method does not exist
    if not isinstance(field, CenteredGrid):
        field = as_center_grid_field(field)
    if len(channel(field)) == 0:
        field = math.expand(field, channel(vector=1))
    if extrap is not None:
        field = field.with_extrapolation(extrap)
    return field


def cat_fields(vs, extrap=None):
    """
    Densely packs a list of fields into a single vector CenteredGrid along the channel dimension
    """
    return field.concat([
        as_cattable_field(v, extrap=extrap)
        for v in vs
    ], dim=channel('vector'))


def copy_field(field: Field) -> Field:
    '''
    Return a (deep) copy of the input field.
    Surely there is a simpler way?
    '''
    return field.__class__(
        values=math.copy(field.values),
        bounds=field.bounds,
        resolution = field.resolution,
        extrapolation = field.extrapolation)


def corruptor(
        in_p_var=0.0, in_v_var=0.0,
        out_p_var=0.0, out_v_var=0.0,
        f_var=0.0):
    """
    factory returns a function to corrupt observations with Gaussian noise.
    This is for observations only; the process noise is handled elsewhere.
    """

    def corrupt(seq):
        """
        last_n plus multi_sim_context_generator return sims in a the ordering
        [(particle, velocity, pressure), (particle, velocity, pressure), ... force ].
        """
        new_seq = []

        force = seq.pop()
        final = seq.pop()

        for i, obs in enumerate(seq):
            new_obs = []
            new_obs.append(obs[0] + math.random_normal(obs[0].shape) * in_p_var**0.5)
            new_obs.append(obs[1] + math.random_normal(obs[1].shape) * in_v_var**0.5)
            new_obs.extend(obs[2:])
            new_seq.append(tuple(new_obs))
        
        new_obs = []
        new_obs.append(final[0] + math.random_normal(final[0].shape) * out_p_var**0.5)
        new_obs.append(final[1] + math.random_normal(final[1].shape) * out_v_var**0.5)
        new_obs.extend(final[2:])
        new_seq.append(tuple(new_obs))

        new_seq.append(force + math.random_normal(force.shape) * f_var**0.5)

        return new_seq

    return corrupt


def pvf_p_stacker(seq):
    """
    last_n plus multi_sim_context_generator return sims in a the ordering
    [(particle, velocity, pressure), (particle, velocity, pressure), ... force ].
    We need (X,y) pairs to train the model.
    This one stacks them down ((force, particle, velocity, particle, velocity,...) particle)
    """
    force = seq.pop()
    final = seq.pop()
    return (
        cat_fields(
            list(
                chain(*[step[0:2] for step in seq])
            ) + [force],
            extrap=extrapolation.NONE
        ),
        # formatting.cat_fields([final[0]]) # this would expand the channel dim
        final[0]
    )

def ppf_p_stacker(seq):
    """
    last_n plus multi_sim_context_generator return sims in a weird ordering.
    [(particle, velocity, pressure), (particle, velocity, pressure), ... force ].
    We need (X,y) pairs to train the model.
    This one stacks them down ((force, particle, particle,...) particle).
    Untested.
    """
    raise NotImplementedError
    force = seq.pop()
    final = seq.pop()
    yield (
        cat_fields(
            [force] + [step[0] for step in seq],
            extrap=extrapolation.NONE
        ),
        final[0]
    )


def pvf_pv_stacker(seq):
    """
    Modify pvf_p_stacker to also deliver a velocity field.
    """
    force = seq.pop()
    final = seq.pop()
    return (
        cat_fields(
            list(
                chain(*[step[0:2] for step in seq])
            ) + [force],
            extrap=extrapolation.NONE
        ),
        # final[0], final[1] # old version; has since been replaced!!!
        cat_fields([final[0], final[1]], extrap=extrapolation.NONE)
    )

    
Torchified = namedtuple(
    'Torchified',
    ['init_rand_t',  'simulate_t',  'p_wrapper',  'v_wrapper',  'f_wrapper'])


def pre_curry(func, *args):
    """
    curries a sim function args, returns first arg
    """
    def _curried(x):
        return func(*args, x)[0]
    return _curried


def torchify(init_rand, simulate, grid_size_x, grid_size_y):
    """
    automatically translate the phiflow predictors into pytorch tensors.
    It would be more efficient if we did not instantiate here to check dims, but this is a one-time cost.
    """

    particle, velocity, force = init_rand(n_batch=0)

    p_wrapper = TorchVectorWrapper(
        particle,
        x=grid_size_x,
        y=grid_size_y
    )
    v_wrapper = TorchVectorWrapper(
        velocity,
        x=grid_size_x,
        y=grid_size_y,
        vector=2,
    )
    f_wrapper = TorchVectorWrapper(
        force,
        x=grid_size_x,
        y=grid_size_y,
        vector=2,
    )

    def init_rand_t(n_batch):
        """
        the init function for phiflow, but as a pytorch tensor factory
        """
        particle, velocity, force = init_rand(n_batch)
        return p_wrapper.unwrap(particle), v_wrapper.unwrap(velocity), f_wrapper.unwrap(force)
    
    def simulate_t(particle_t, velocity_t, force_t, *args, **kwargs):
        """
        the simulate function for phiflow, but as a pytorch tensor predictor
        """
        pred_particle, pred_velocity, pressure = simulate(
            p_wrapper.wrap(particle_t),
            v_wrapper.wrap(velocity_t),
            f_wrapper.wrap(force_t), *args, **kwargs)
        return p_wrapper.unwrap(pred_particle), v_wrapper.unwrap(pred_velocity), pressure

    return Torchified(init_rand_t, simulate_t, p_wrapper, v_wrapper, f_wrapper)
