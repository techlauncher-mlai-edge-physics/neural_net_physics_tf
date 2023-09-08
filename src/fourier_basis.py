from functools import reduce
import tensorflow as tf
from . import formatting, iterators
from einops import rearrange
import numpy as np
from operator import mul


def phase(
        dim_sizes,
        pos_low: float = -1,
        pos_high: float = 1,
        centered: bool = True,
        dtype=tf.float32,
        device=None):
    """
    Basically a clone of `encode_positions`,
    but reimplemented here to keep the other thing unpickling correctly.
    """
    def generate_grid(size):
        width = (pos_high - pos_low) / size
        if centered:
            start = pos_low + width/2
            stop = pos_high - width/2
        else:
            start = pos_low
            stop = pos_high - width
        return tf.linspace(
            start, stop, num=size)
    grid_list = list(map(generate_grid, dim_sizes))
    grid = tf.stack(tf.meshgrid(*grid_list, indexing='ij'), axis=-1)
    # # stack never fails to increase the number of dimensions, which is not what we want for univariate grids
    # if len(dim_sizes) == 1:
    #     grid = grid.squeeze(-1)
    return grid


def basis(
    grid, fs_even, fs_odd=None,
    norm=True  # Normalise to be "unitary" in the sense that the inner product is 1
):
    """
    map a list of frequency tuples over a list of grids,
    return cos and sin bases using these frequencies.
    Not an efficient way of handling DC.
    untested for d>2.

    Example:
    >>> grid = phase((10, 10))
    >>> fs = tf.constant([[1, 0], [0, 1]], dtype=tf.float32)
    >>> basis(grid, *fs)
    """
    if fs_odd is None:
        # really this codepath is for convenience of testing only.
        fs_odd = fs_even
    # hack because it feels weird to return fs with a dim index in 1d
    if fs_even.ndim == 1:
        fs_even = tf.expand_dims(fs_even, axis=-1)
    if fs_odd.ndim == 1:
        fs_odd = tf.expand_dims(fs_odd, axis=-1)
    phases_even = tf.einsum("...i,ki->k...", grid, fs_even) * \
        tf.constant(np.pi, dtype=tf.float32)
    phases_odd = tf.einsum("...i,ki->k...", grid, fs_odd) * \
        tf.constant(np.pi, dtype=tf.float32)
    # Calling trig functions profligately assuming that this will be cached
    # basis = tf.concat([tf.cos(phases_even), tf.sin(phases_odd)], dim=0)
    basis_even = tf.cos(phases_even)
    basis_odd = tf.sin(phases_odd)
    scale = reduce(mul, grid.shape[:-1])  # last axis is packed with dims
    # print("scale", grid.shape[:-1], sqrt(scale))
    if norm:
        basis_even /= (tf.sqrt(scale/2))
        basis_odd /= (tf.sqrt(scale/2))
    return basis_even, basis_odd


def basis_flat(
    *args, **kwargs
):
    """
    same as `basis` but interleaves sine and cos terms.

    Example:
    >>> grid = phase((10, 10))
    >>> fs = tf.constant([[1, 0], [0, 1]], dtype=tf.float32)
    >>> basis_flat(grid, *fs)
    """
    return tf.concat(basis(*args, **kwargs), axis=0)


def complete_fs(
        dim_ranges, dc=False,
        dtype=tf.float32,
        device=None, sorted=False):
    """
    Generate a complete list of frequency tuples for given set of spectral ranges, excluding linear combinations.
    User is responsible for avoiding aliasing.
    Negative frequencies in the first dimension are ignored.

    Example:
    >>> phases = fourier_basis.phase((16,16))
    >>> fs = fourier_basis.complete_fs((3,3), dc=True)
    >>> bases = fourier_basis.basis(phases, *fs, norm=False)
    """
    # When the first axis is 0 we have some extra symmetries
    # I THINK this handles those; have not thoroughly tested in d>2
    dim_ranges = list(dim_ranges)
    # We allow dim_ranges to be tuples so that we can be asymmetric;
    # this only makes sense for (low, high) pairs if abs(low+high) <=1
    for i, r in enumerate(dim_ranges):
        if isinstance(r, (float, int)):
            if i > 0:
                dim_ranges[i] = (-r, r)
            else:
                # First axis is special because of symmetry of real transform
                dim_ranges[i] = (0, r)

    # range_even = [tf.zeros((1,), dtype=dtype)]
    # range_odd = [tf.range(1, dim_ranges[0][1]+1, dtype=dtype)]
    range_even = [tf.range(0, dim_ranges[0][1]+1, dtype=dtype)]
    range_odd = [tf.range(1, dim_ranges[0][1]+1, dtype=dtype)]
    for l, h in dim_ranges[1:]:
        range_even.append(
            tf.range(0, h+1, dtype=dtype))
        range_odd.append(
            tf.range(l, h+1, dtype=dtype))
    # print("range_even", range_even)
    # print("range_odd", range_odd)

    fs_even_l, fs_even_r = tf.meshgrid(*range_even, indexing='ij')
    fs_even = tf.stack([tf.reshape(fs_even_l, [-1]),
                       tf.reshape(fs_even_r, [-1])], axis=1)
    fs_odd_l, fs_odd_r = tf.meshgrid(*range_odd, indexing='ij')
    fs_odd = tf.stack([tf.reshape(fs_odd_l, [-1]),
                      tf.reshape(fs_odd_r, [-1])], axis=1)

    # print(fs_even.shape, fs_odd.shape)

    if not dc:
        fs_even = fs_even[1:]

    # we usually introduce the bias term elsewhere and anyway sin(0)=0
    # sort by frequency
    if sorted:
        fs_even = tf.gather(fs_even, tf.argsort(
            tf.linalg.norm(fs_even, axis=-1)))
        fs_odd = tf.gather(fs_odd, tf.argsort(tf.linalg.norm(fs_odd, axis=-1)))
    return (fs_even, fs_odd)
