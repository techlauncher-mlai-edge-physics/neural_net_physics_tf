# %%
from functools import partial
from einops import rearrange
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
class FourierLayer2dLite(tf.keras.layers.Layer):
    def __init__(self, in_dim, out_dim, n_modes):
        super(FourierLayer2d, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes
        self.linear = tf.keras.layers.Dense(out_dim)
        self.act = tf.keras.layers.ReLU()

        # fourier_weight = [
        #     tf.Variable(
        #         tf.random.normal(
        #             shape=(in_dim, out_dim, n_modes, n_modes, 2),
        #             stddev=1 / (in_dim * out_dim),
        #         )
        #     )
        #     for _ in range(2)
        # ]
        # self.fourier_weight = fourier_weight

        # Create the first variable using add_weight
        self.fourier_weight_1 = self.add_weight(
            shape=(in_dim, out_dim, n_modes, n_modes, 2),
            initializer="glorot_uniform",
            trainable=True,
            name="FourierLayer2d/fourier_weight_1",
        )

        # Create the second variable using add_weight
        self.fourier_weight_2 = self.add_weight(
            shape=(in_dim, out_dim, n_modes, n_modes, 2),
            initializer="glorot_uniform",
            trainable=True,
            name="FourierLayer2d/fourier_weight_2",
        )

    @staticmethod
    def complex_matmul_2d(a, b):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        # assert they have the dimension
        op = partial(tf.einsum, "bixy,ioxy->boxy")
        return tf.stack(
            [
                op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
                op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1]),
            ],
            axis=-1,
        )

    def call(self, inputs):
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        B, M, N, I = inputs.shape

        inputs = tf.transpose(inputs, perm=[0, 3, 1, 2])
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        x_ft = tf.signal.rfft2d(inputs, fft_length=[M, N])
        scale = tf.sqrt(tf.cast(tf.reduce_prod(tf.shape(inputs)[-2:]), tf.complex64))
        x_ft /= scale

        x_ft = tf.stack([tf.math.real(x_ft), tf.math.imag(x_ft)], axis=-1)
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft = self.complex_matmul_2d(
            x_ft[:, :, : self.n_modes, : self.n_modes], self.fourier_weight_1
        )
        # out_ft.shape = (batch_size, in_dim, n_modes, n_modes, 2)
        out_ft_zero = tf.zeros(
            [B, I, N - self.n_modes * 2, self.n_modes, 2], dtype=tf.float32
        )
        # out_ft_zero.shape = (batch_size, in_dim, grid_size - n_modes * 2, n_modes, 2)
        out_ft = tf.concat([out_ft, out_ft_zero], axis=-3)
        # out_ft.shape = (batch_size, in_dim, grid_size - n_modes, n_modes, 2)
        out_ft = tf.concat(
            [
                out_ft,
                self.complex_matmul_2d(
                    x_ft[:, :, : self.n_modes, : self.n_modes], self.fourier_weight_2
                ),
            ],
            axis=-3,
        )
        # out_ft.shape = (batch_size, in_dim, grid_size , n_modes, 2)
        out_ft = tf.concat(
            [out_ft, tf.zeros([B, I, N, M // 2 + 1 - self.n_modes, 2])], axis=-2
        )
        # out_ft.shape = (batch_size, in_dim, grid_size , grid_size // 2 + 1, 2)
        out_ft = tf.complex(out_ft[..., 0], out_ft[..., 1])

        inputs = tf.signal.irfft2d(out_ft, fft_length=[N, M])
        scale = tf.sqrt(tf.cast(tf.reduce_prod(tf.shape(out_ft)[-2:]), tf.float32))
        inputs /= scale

        inputs = tf.transpose(inputs, perm=[0, 2, 3, 1])

        res = self.linear(inputs)
        res = self.act(res + inputs)
        return res


# %%
class FourierLayer2d(tf.keras.layers.Layer):
    def __init__(self, in_dim, out_dim, n_modes, residual=True):
        super(FourierLayer2d, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes
        self.linear = tf.keras.layers.Dense(out_dim)
        self.residual = residual
        self.act = tf.keras.layers.ReLU()

        fourier_weight = [
            tf.Variable(
                tf.random.normal(
                    shape=(in_dim, out_dim, n_modes, n_modes, 2),
                    stddev=1 / (in_dim * out_dim),
                )
            )
            for _ in range(2)
        ]
        self.fourier_weight = fourier_weight

    @staticmethod
    def complex_matmul_2d(a, b):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        # assert they have the dimension
        op = partial(tf.einsum, "bixy,ioxy->boxy")
        return tf.stack(
            [
                op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
                op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1]),
            ],
            axis=-1,
        )

    def call(self, x):
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        B, M, N, I = x.shape

        x = tf.transpose(x, perm=[0, 3, 1, 2])
        assert x.dtype == tf.float32
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # x_ft_real = tf.signal.rfft(x, fft_length=M, name="rfft_real")
        # x_ft_imag = tf.signal.rfft(tf.reverse(x, axis=[-1]), fft_length=M, name="rfft_imag")
        # x_ft_imag = tf.reverse(x_ft_imag, axis=[-1])

        x_ft = tf.signal.rfft2d(x, fft_length=[M, N])

        x_ft = tf.stack([tf.math.real(x_ft), tf.math.imag(x_ft)], axis=-1)

        # downcast to float32, see https://www.tensorflow.org/api_docs/python/tf/signal/rfft2d, since tf.signal.rfft2d returns complex64 and input is float32
        x_ft = tf.cast(x_ft, tf.float32)

        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        # x_ft_stacked = tf.stack([x_ft_real, x_ft_imag], axis=-1)
        # x_ft_stacked.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]
        out_ft = self.complex_matmul_2d(
            x_ft[:, :, : self.n_modes, : self.n_modes], self.fourier_weight[0]
        )
        # out_ft.shape=(batch_size, in_dim, n_modes, n_modes, 2)
        out_ft_zero = tf.zeros(
            [B, I, N - self.n_modes * 2, self.n_modes, 2], dtype=tf.float32
        )
        out_ft = tf.concat([out_ft, out_ft_zero], axis=-3)
        out_ft = tf.concat(
            [
                out_ft,
                self.complex_matmul_2d(
                    x_ft[:, :, : self.n_modes, : self.n_modes], self.fourier_weight[1]
                ),
            ],
            axis=-3,
        )
        out_ft = tf.concat(
            [out_ft, tf.zeros([B, I, N, M // 2 + 1 - self.n_modes, 2])], axis=-2
        )

        out_ft = tf.complex(out_ft[..., 0], out_ft[..., 1])

        x = tf.signal.irfft2d(out_ft, fft_length=[N, M])

        # x_ift_real = tf.signal.irfft(out_ft, fft_length=M, name="irfft_real")
        # x_ift_real.shape == [batch_size, grid_size, grid_size, in_dim]

        x = tf.transpose(x, perm=[0, 2, 3, 1])

        if self.residual:
            res = self.linear(x)
            x = self.act(x + res)
        else:
            x = self.act(x)
        return x


# %%
class FNO2d(tf.keras.layers.Layer):
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
