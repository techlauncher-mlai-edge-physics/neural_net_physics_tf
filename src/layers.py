# %%
from functools import partial
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
        super(FourierLayer2dLite, self).__init__()
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
            name="FourierLayer2dLite/fourier_weight_1",
        )

        # Create the second variable using add_weight
        self.fourier_weight_2 = self.add_weight(
            shape=(in_dim, out_dim, n_modes, n_modes, 2),
            initializer="glorot_uniform",
            trainable=True,
            name="FourierLayer2dLite/fourier_weight_2",
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
        _, M, N, I = inputs.shape
        dynamic_shape = tf.shape(inputs)

        inputs = tf.transpose(inputs, perm=[0, 3, 1, 2])
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        x_ft = tf.signal.rfft2d(inputs, fft_length=[M, N])
        scale = tf.sqrt(tf.cast(tf.reduce_prod(dynamic_shape[-2:]), tf.complex64))
        x_ft /= scale

        x_ft = tf.stack([tf.math.real(x_ft), tf.math.imag(x_ft)], axis=4)
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft = self.complex_matmul_2d(
            x_ft[:, :, : self.n_modes, : self.n_modes], self.fourier_weight_1
        )
        # out_ft.shape = (batch_size, in_dim, n_modes, n_modes, 2)
        # dynamic shape
        out_ft_zero = tf.zeros(
            [dynamic_shape[0], I, N - self.n_modes * 2, self.n_modes, 2],
            dtype=tf.float32,
        )
        # out_ft_zero.shape = (batch_size, in_dim, grid_size - n_modes * 2, n_modes, 2)
        out_ft = tf.concat([out_ft, out_ft_zero], axis=-3)
        # out_ft.shape = (batch_size, in_dim, grid_size - n_modes, n_modes, 2)
        out_ft = tf.concat(
            [
                out_ft,
                self.complex_matmul_2d(
                    x_ft[:, :, -self.n_modes :, : self.n_modes], self.fourier_weight_2
                ),
            ],
            axis=-3,
        )
        # out_ft.shape = (batch_size, in_dim, grid_size , n_modes, 2)
        out_ft = tf.concat(
            [out_ft, tf.zeros([dynamic_shape[0], I, N, M // 2 + 1 - self.n_modes, 2])],
            axis=-2,
        )
        # out_ft.shape = (batch_size, in_dim, grid_size , grid_size // 2 + 1, 2)
        out_ft = tf.complex(out_ft[..., 0], out_ft[..., 1])

        inputs = tf.signal.irfft2d(out_ft, fft_length=[N, M])
        scale = tf.sqrt(tf.cast(tf.reduce_prod(tf.shape(out_ft)[-2:]), tf.float32))
        inputs *= scale

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
        _, M, N, I = x.shape
        dynamic_shape = tf.shape(x)

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
            [dynamic_shape[0], I, N - self.n_modes * 2, self.n_modes, 2],
            dtype=tf.float32,
        )
        out_ft = tf.concat([out_ft, out_ft_zero], axis=-3)
        out_ft = tf.concat(
            [
                out_ft,
                self.complex_matmul_2d(
                    x_ft[:, :, -self.n_modes :, : self.n_modes], self.fourier_weight[1]
                ),
            ],
            axis=-3,
        )
        out_ft = tf.concat(
            [out_ft, tf.zeros([dynamic_shape[0], I, N, M // 2 + 1 - self.n_modes, 2])],
            axis=-2,
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
