# %%
import bz2
import datetime
import gc  # garbage collector for training with constrained
import os

import dill
import matplotlib.pyplot as plt
import tensorflow as tf
from einops import rearrange
from matplotlib_inline.backend_inline import set_matplotlib_formats
from phi import math, vis
from phi.tf import TENSORFLOW
from phi.tf.flow import *
from tqdm import tqdm

from src import formatting, fourier_basis, iterators, physical_models, plots
from src.formatting import pvf_pv_stacker
from src.models import fno_2d
from src.sim_iterator import simple_sim_gen

set_matplotlib_formats(
    "jpeg",
)
PHI_DEVICE = "CPU"

set_matplotlib_formats(
    "jpeg",
)  # <- reduce ipynb size
# %%
grid_size = (64, 64)
grid_size_x, grid_size_y = grid_size
DT = 0.01
v_noise_power = 0.0
n_steps = 5
max_steps = 1000
n_batch = 64
decay_rate = 6e-7
initial_lr = 1e-3
n_skip_steps = 5

init_rand, sim_step = physical_models.ns_sim(
    phi_device=PHI_DEVICE,
    grid_size=grid_size,
    jit=False,
    incomp=False,
    v_noise_power=1e6,
    backend="tensorflow",
    DT=DT/n_skip_steps,
    n_skip_steps=n_skip_steps,
)

# %%

# truth
math.seed(42)
particle, velocity, force = init_rand(n_batch=1)
vis.plot(particle.batch[0], show_color_bar=False)
vis.plot(velocity.batch[0] * 0.01)

# %%
math.seed(989)


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

sd_particle, sd_velocity, sd_force

# %%
phases = fourier_basis.phase(grid_size)
fs = fourier_basis.complete_fs((12, 12), dc=False)
bases = fourier_basis.basis_flat(phases, *fs, norm=True)
# %%
model = fno_2d(
    in_channels=5,
    out_channels=3,
    width=24,
    n_layers=5,
    nearly_last_width=64
)

# %%
model(
    *formatting.to_native_chan_last(
        particle, velocity, force
    )
)
# %%
losses = []
LOG_PATH = "debug"

SUBLOG_PATH = os.path.join(
    LOG_PATH, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# writer = SummaryWriter(log_dir=SUBLOG_PATH)


def num_params(model):
    return sum(param.numel() for param in model.parameters())


#
dl = simple_sim_gen(
    init_rand,
    sim_step,
    n_steps=n_steps,
    n_context=2,
    max_steps=max_steps,
    n_batch=n_batch,
    # in_p_var=0.01,
    # in_v_var=0.01,
    # out_p_var=0.01,
    # out_v_var=0.01,
    # f_var=0.01,
    stacker=pvf_pv_stacker,
)

optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr, weight_decay=decay_rate)
loss_function = tf.keras.losses.MeanSquaredError()

try:
    for step, (X, y) in enumerate(tqdm(dl)):
        X_particle = X[:, :, :, :1] / sd_particle
        X_velocity = X[:, :, :, 1:3] / sd_velocity
        X_force = X[:, :, :, 3:] / sd_force
        y_particle = y[:, :, :, :1] / sd_particle
        y_velocity = y[:, :, :, 1:3] / sd_velocity

        X = tf.concat([X_particle, X_velocity, X_force], axis=-1)
        y = tf.concat([y_particle, y_velocity], axis=-1)
        with tf.GradientTape() as tape:
            pred = model(X, training=True)
            loss = loss_function(y, pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        losses.append(loss.numpy())
        if step % 13 == 0:
            print(f"iter={step}, MSE_for_this_iter={loss}")
            im = plots.pred_actual_plot(
                pred[0, :, :, 0].numpy(),
                y[0, :, :, 0].numpy()
            )
            plt.show()
            # # write an image of the output to tensorboard:

finally:
    pass

# %%
MODEL_NAME = "fno2d_001"
dill.dump(model, file=bz2.open(f"models/{MODEL_NAME}.pth.bz2", "wb"))


# %%
@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(1, grid_size_x, grid_size_y, 5), dtype=tf.float32)
    ]
)
def fno2_predict(x):
    return {"outputs": model(x)}


module = tf.Module()
module.serve = fno2_predict
module.model = model


tf.saved_model.save(
    module, f"models/{MODEL_NAME}", signatures={"serving_default": module.serve}
)

# %%
