# %%
%load_ext autoreload
%autoreload 2
# %%
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib_inline.backend_inline import set_matplotlib_formats
from phi import math, vis
from phi.tf import TENSORFLOW
from phi.tf.flow import *
from tqdm import tqdm

from src import formatting, fourier_basis, iterators, physical_models, plots
from src.sim_iterator import simple_sim_gen
# 1. Import TensorBoard
from tensorflow.summary import create_file_writer

# 2. Create a file writer object. You can specify a log directory of your choice. For example, 'logs/'

run_name = "dan_exp_002"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f'logs/{run_name}_{current_time}'
file_writer = create_file_writer(log_dir)

set_matplotlib_formats(
    "jpeg",
)
PHI_DEVICE = "CPU"

set_matplotlib_formats(
    "jpeg",
)  # <- reduce ipynb size
PHI_DEVICE = "CPU"
# %%
grid_size = (64, 64)
grid_size_x, grid_size_y = grid_size
n_batch = 1

init_rand, sim_step = physical_models.shallow_water_sim(
    phi_device=PHI_DEVICE,
    grid_size=grid_size,
    jit=False,
    # v_noise_power=1e6,
    backend="tensorflow",
    scale=0.1,
    force_scale=0.0,
    velocity_scale=0.0,
    n_blob=2,
    gravity=10,
    DT=0.01
)

math.seed(42)
height, velocity, force = init_rand(n_batch=n_batch)
vis.plot(velocity * 0.05, title="velocity")
plt.close()
vis.plot(height, show_color_bar=False, title="height")
plt.close()

pressure = None

for _ in range(10):
    height, velocity, pressure = sim_step(
        height, velocity, force, pressure)
    vis.plot(velocity * 0.05, title="velocity")
    vis.plot(height, show_color_bar=False, title="height")

# %%
