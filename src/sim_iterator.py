import sys
import tensorflow as tf
from . import formatting, iterators
from einops import rearrange


def simple_sim_gen(
        init_rand, sim_step,
        n_steps=5,
        n_context=2,
        n_batch=8,
        max_steps=sys.maxsize,
        stacker=formatting.pvf_p_stacker,
        in_v_var=0.0,
        in_p_var=0.0,
        out_p_var=0.0,
        out_v_var=0.0,
        f_var=0.0,
        *args, **kwargs):
    it = iterators.multi_sim_context_generator(
        init_rand,
        sim_step,
        n_steps=n_steps,
        n_context=n_context,
        max_steps=max_steps,
        n_batch=n_batch,
        *args, **kwargs
    )
    if in_v_var or in_p_var or out_p_var or out_v_var or f_var:
        it = iterators.corrupt(
            it,
            in_v_var=in_v_var,
            in_p_var=in_p_var,
            out_p_var=out_p_var,
            out_v_var=out_v_var,
            f_var=f_var
        )
    return map(
        formatting.to_natives_chan_last,
        map(stacker, it)
    )


def batch_flat_y(data):
    X, y = data
    return X, rearrange(y, 'b x y c -> (b x y) c')


def vector_flat_y(data):
    X, y = data
    return X, rearrange(y, 'b x y c -> b (x y) c')
