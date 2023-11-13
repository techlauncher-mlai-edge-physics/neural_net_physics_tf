# %%
import datetime
import gc  # garbage collector for training with constrained
import os

import tensorflow as tf
from einops import rearrange
from matplotlib_inline.backend_inline import set_matplotlib_formats
from phi import math

# import phi.tf.flow as phiflow

from src import formatting, fourier_basis, physical_models
from src.formatting import pvf_pv_stacker
from src.models import fno_2d, fno_2d_lite, fno_2d_Seq
from src.sim_iterator import simple_sim_gen

set_matplotlib_formats(
    "jpeg",
)  # <- reduce ipynb size
PHI_DEVICE = "CPU"
# %%
grid_size = (64, 64)
grid_size_x, grid_size_y = grid_size

init_rand, sim_step = physical_models.ns_sim(
    phi_device=PHI_DEVICE,
    grid_size=grid_size,
    jit=False,
    incomp=False,
    v_noise_power=1e6,
    backend="cpu",
)

# %%


def simulate(particle, velocity, force, n_skip_steps=1):
    pressure = None
    for _ in range(n_skip_steps):
        particle, velocity, pressure = sim_step(particle, velocity, force, pressure)
    return particle, velocity, pressure


# truth
math.seed(42)
particle, velocity, force = init_rand(n_batch=1)

# %%
math.seed(989)

max_steps = 1600


dl = simple_sim_gen(
    init_rand,
    sim_step,
    n_steps=5,
    n_context=2,
    max_steps=max_steps,
    n_batch=200,
    in_p_var=0.01,
    in_v_var=0.1,
    out_p_var=0.01,
    out_v_var=0.1,
    f_var=0.01,
    stacker=pvf_pv_stacker,
)

# %%
X, y = next(iter(dl))

X_particle = X[:, :, :, 0]
X_particle = rearrange(X_particle, "b x y -> (b x y)")
X_velocity = X[:, :, :, 1:3]
X_velocity = rearrange(X_velocity, "b x y c -> (b x y c)")
X_force = X[:, :, :, 3:]
X_force = rearrange(X_force, "b x y c -> (b x y c)")

# %%
sd_particle = tf.math.reduce_std(X_particle).numpy()
sd_velocity = tf.math.reduce_std(X_velocity).numpy()
sd_force = tf.math.reduce_std(X_force).numpy()

del X, y, dl, X_particle, X_velocity, X_force
gc.collect()
# torch.cuda.empty_cache()

print(sd_particle, sd_velocity, sd_force)

# %%
phases = fourier_basis.phase(grid_size)
fs = fourier_basis.complete_fs((12, 12), dc=False)
bases = fourier_basis.basis_flat(phases, *fs, norm=True)
# %%
model = fno_2d_lite(
    in_channels=5, out_channels=3, width=13, modes=8, n_layers=2, nearly_last_width=146
)

# %%
model(tf.zeros((1, 64, 64, 5)))
# %%
v_noise_power = 1e6
n_steps = 5
max_steps = 2000
n_batch = 32
lr = 1e-3
weight_decay = 6e-7


LOG_PATH = "debug"

SUBLOG_PATH = os.path.join(LOG_PATH, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

dl = simple_sim_gen(
    init_rand,
    sim_step,
    n_steps=n_steps,
    n_context=2,
    max_steps=max_steps,
    n_batch=n_batch,
    stacker=pvf_pv_stacker,
)


def normalize(X, y):
    X_particle = X[:, :, :, :1] / sd_particle
    X_velocity = X[:, :, :, 1:3] / sd_velocity
    X_force = X[:, :, :, 3:] / sd_force
    y_particle = y[:, :, :, :1] / sd_particle
    y_velocity = y[:, :, :, 1:3] / sd_velocity
    X = tf.concat([X_particle, X_velocity, X_force], axis=-1)
    y = tf.concat([y_particle, y_velocity], axis=-1)
    return X, y


# Convert the simulation data to a TensorFlow dataset
dataset = tf.data.Dataset.from_generator(
    lambda: (normalize(X, y) for X, y in dl),
    output_types=(tf.float32, tf.float32),
    output_shapes=((n_batch, 64, 64, 5), (n_batch, 64, 64, 3)),
)

model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=lr, weight_decay=weight_decay, epsilon=1e-7
    ),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=["mean_squared_error"],
)

print(model.summary())

# tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)

callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir=SUBLOG_PATH, histogram_freq=1, write_graph=True, write_images=True
    )
]

dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset = dataset.repeat()

model.fit(
    dataset,
    epochs=1,
    steps_per_epoch=2,
    callbacks=callbacks,
    verbose=1,
)

# save the model
model_name = model.__class__.__name__
model.save(os.path.join("models", model_name + ".keras"))
model.save(os.path.join("models", model_name), save_format="tf")

# %%
