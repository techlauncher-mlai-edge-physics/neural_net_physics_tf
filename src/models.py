# %%
import tensorflow as tf
from einops import rearrange
from src.layers import FourierLayer2d, FourierLayer2dLite


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
        **kwargs,
    ):
        super().__init__(**kwargs)

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

        self.input_dim = in_channels
        w = self.width[0]
        self.in_proj = tf.keras.layers.Dense(w)
        self.spectral_layers = []
        for next_w, m in zip(self.width[1:], modes):
            self.spectral_layers.append(
                FourierLayer2dLite(
                    in_dim=w,
                    out_dim=next_w,
                    n_modes=m,
                )
            )
            w = next_w
        self.out_1 = tf.keras.layers.Dense(nearly_last_width, use_bias=bias_1)
        w = nearly_last_width
        self.out_act = tf.keras.layers.ReLU()
        self.out_2 = tf.keras.layers.Dense(self.out_channels, use_bias=bias_2)

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


def fno_2d_Seq(
    in_channels=3,
    out_channels=1,
    modes=17,
    width=20,
    n_layers=4,
    nearly_last_width=128,
    bias_1=True,
    bias_2=True,
):
    if isinstance(modes, int):
        modes = [modes] * n_layers
        assert len(modes) == n_layers
    if isinstance(width, int):
        width = [width] * (n_layers + 1)
    w = width[0]
    in_proj = tf.keras.layers.Dense(w)
    spectral_layers = []
    for next_w, m in zip(width[1:], modes):
        spectral_layers.append(
            FourierLayer2dLite(
                in_dim=w,
                out_dim=next_w,
                n_modes=m,
            )
        )
        w = next_w
    out_1 = tf.keras.layers.Dense(nearly_last_width, use_bias=bias_1)
    w = nearly_last_width
    out_act = tf.keras.layers.ReLU()
    out_2 = tf.keras.layers.Dense(out_channels, use_bias=bias_2)
    return tf.keras.Sequential(
        [in_proj, *spectral_layers, out_1, out_act, out_2]
    )

# %%
class FNO2d(tf.keras.Model):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        modes=17,
        width=20,
        n_layers=4,
        residual=False,
        conv_residual=True,
        nearly_last_width=128,
        pos_low=-1.0,
        pos_high=1.0,
        flat_mode=False,
        bias_1=True,
        bias_2=True,
    ):
        super(FNO2d, self).__init__()
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

        self.residual = residual
        self.pos_low = pos_low
        self.pos_high = pos_high

        self.input_dim = in_channels + spatial_dim
        w = self.width[0]
        self.in_proj = tf.keras.layers.Dense(w)
        self.spectral_layers = []
        for next_w, m in zip(self.width[1:], modes):
            self.spectral_layers.append(
                FourierLayer2d(
                    in_dim=w,
                    out_dim=next_w,
                    n_modes=m,
                    residual=conv_residual,
                )
            )
            w = next_w
        if nearly_last_width > 0:
            self.out_1 = tf.keras.layers.Dense(nearly_last_width, use_bias=bias_1)
            w = nearly_last_width
            self.out_act = tf.keras.layers.ReLU()
        else:
            self.out_1 = tf.keras.layers.Identity()
            self.out_act = tf.keras.layers.Identity()

        self.out_2 = tf.keras.layers.Dense(self.out_channels, use_bias=bias_2)
        self.flat_mode = flat_mode

    def call(self, *predictors):
        x = self.last_layer(*predictors)
        x = self.out_2(x)
        if self.flat_mode == "batch":
            x = rearrange(x, "b x y c -> (b x y) c")
        elif self.flat_mode == "vector":
            x = rearrange(x, "b x y c -> b (x y) c")
        return x

    def last_layer(self, *predictors):
        x = self._build_features(*predictors)
        x = self.in_proj(x)
        for layer in self.spectral_layers:
            x = layer(x) + x if self.residual else layer(x)
        x = self.out_1(x)
        return self.out_act(x)

    def _encode_positions(self, dim_sizes):
        return phase(dim_sizes=dim_sizes, pos_low=self.pos_low, pos_high=self.pos_high)

    def _build_features(self, *predictors):
        # check what the predictors' type is
        B, *dim_sizes, T = predictors[0].shape
        pos_feats = self._encode_positions(dim_sizes)
        pos_feats = tf.repeat(pos_feats[tf.newaxis, ...], B, axis=0)
        predictor_arr = tf.concat(predictors + (pos_feats,), axis=-1)
        return predictor_arr


# %%
def fno_2d(*args, **kwargs):
    net = FNO2d(*args, **kwargs)
    # net.build(input_shape=(None, None, None, kwargs.get('in_channels', 3)))  # Adjust input shape
    return net