# %%
from einops import rearrange
from layers import FourierLayer2d, FourierLayer2dLite

import tensorflow as tf

# %%
def phase(
    dim_sizes,
    pos_low: float = -1,
    pos_high: float = 1,
    centered: bool = True,
    dtype=tf.float32,
):
    """
    Basically a clone of `encode_positions`,
    but reimplemented here to keep the other thing unpickling correctly.
    """

    def generate_grid(size):
        width = (pos_high - pos_low) / size
        if centered:
            left = pos_low + width / 2
            right = pos_high - width / 2
        else:
            left = pos_low
            right = pos_high - width
        return tf.linspace(left, right, num=size)

    grid_list = list(map(generate_grid, dim_sizes))
    grid = tf.stack(tf.meshgrid(*grid_list, indexing="ij"), axis=-1)
    return grid


# %%
class FNO2dLiteModel(tf.keras.Model):
    def __init__(
            self,
            in_channels=3,
            out_channels=1,
            modes=17,
            width=20,
            n_layers=4,
            nearly_last_width=128,
            pos_low=-1.0,
            pos_high=1.0,
            bias_1=True,
            bias_2=True,
            **kwargs):
        super().__init__(**kwargs)
        spatial_dim = 2

        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(modes, int):
            modes = [modes] * n_layers
        self.modes = modes
        assert len(self.modes) == n_layers
        if isinstance(width, int):
            width = [width] * (n_layers + 1)
        self.width = width
        assert len(self.width) == n_layers + 1

        self.pos_low = pos_low
        self.pos_high = pos_high

        self.input_dim = (
            in_channels
        )
        w = self.width[0]
        self.in_proj = tf.keras.layers.Dense(w)
        self.spectral_layers = []
        for (next_w, m) in zip(self.width[1:], modes):
            self.spectral_layers.append(FourierLayer2dLite(
                in_dim=w,
                out_dim=next_w,
                n_modes=m,
            ))
            w = next_w
        self.out_1 = tf.keras.layers.Dense(
            nearly_last_width,
            use_bias=bias_1)
        w = nearly_last_width
        self.out_act = tf.keras.layers.ReLU()
        self.out_2 = tf.keras.layers.Dense(
            self.out_channels,
            use_bias=bias_2)
        

    def call(self, *inputs, training=False):
        x = tf.concat(inputs, axis=-1)
        # run the model
        x = self.in_proj(x)
        # x = tf.expand_dims(x, axis=-1)
        for layer in self.spectral_layers:
            x = layer(x)
        # x = tf.squeeze(x, axis=-1)
        x = self.out_1(x)
        x = self.out_act(x)
        x = self.out_2(x)
        return x

# %%
def fno_2d_lite(*args, **kwargs):
    net = FNO2dLiteModel(*args, **kwargs)
    # net.build(input_shape=(None, None, None, kwargs.get('in_channels', 3)))  # Adjust input shape
    return net

def fno_2d_Seq(in_channels=3,
            out_channels=1,
            modes=17,
            width=20,
            n_layers=4,
            nearly_last_width=128,
            bias_1=True,
            bias_2=True):
    if isinstance(modes, int):
        modes = [modes] * n_layers
        assert len(modes) == n_layers
    if isinstance(width, int):
        width = [width] * (n_layers + 1)
    w=width[0]
    in_proj = tf.keras.layers.Dense(w)
    spectral_layers = []
    for (next_w, m) in zip(width[1:], modes):
        spectral_layers.append(FourierLayer2dLite(
            in_dim=w,
            out_dim=next_w,
            n_modes=m,
        ))
        w = next_w
    out_1 = tf.keras.layers.Dense(
        nearly_last_width,
        use_bias=bias_1)
    w = nearly_last_width
    out_act = tf.keras.layers.ReLU()
    out_2 = tf.keras.layers.Dense(
        out_channels,
        use_bias=bias_2)
    concat = tf.keras.layers.Concatenate(axis=-1)
    return tf.keras.Sequential([
        concat,
        in_proj,
        *spectral_layers,
        out_1,
        out_act,
        out_2
    ])
