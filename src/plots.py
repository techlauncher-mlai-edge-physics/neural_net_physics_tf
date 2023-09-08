
import matplotlib.pyplot as plt
from math import sqrt, ceil, floor
import numpy as np
from matplotlib.colors import Normalize
from einops import asnumpy


def multi_img_plot(x, interval=1, n_cols=None, fsize=6, interpolation=None, crange=None):
    """
    Plot array as images, iterating over first axis for each successive one.
    """
    x = asnumpy(x)
    steps = range(0, x.shape[0], interval)
    if n_cols is None:
        n_cols = len(steps)
    print(steps, len(steps), n_cols)
    n_rows = ceil(len(steps) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fsize * n_cols/n_rows, fsize));
    axes = axes.flatten()
    for ax in axes:
        ax.set_axis_off()
    plt.tight_layout()
    if crange is None:
        vmax = max([np.amax(x), -np.amin(x)])
        vmin = -vmax
    elif isinstance(crange, (int, float)):
        vmin = -crange
        vmax = crange
    else:  # tuple, presumably
        vmin, vmax = crange

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('RdBu')
    for i, ax in zip(steps, axes):
        ax.imshow(asnumpy(x[i,...]), interpolation=interpolation, norm=norm, cmap=cmap)
    # fig.colorbar(ims[0], ax=axs, orientation='vertical', shrink = 0.6)
    return fig


def img_plot(x, interval=1, n_cols=None, fsize=6, interpolation=None):
    """
    plot whatever image I can find in x
    """
    x = np.squeeze(x, )
    plt.imshow(x, interpolation=interpolation)
    plt.gca().set_axis_off()
    plt.tight_layout()
    return plt.gcf()


def multi_heatmap(xs, names=None, crange=None, base_size=3.0, dpi=100):
    n_ims = len(xs)
    if names is None:
        names = range(n_ims)
    fig, axs = plt.subplots(1, n_ims, figsize=(base_size*n_ims, base_size), dpi=dpi)
    if crange is None:
        vmax = max([np.abs(x).max() for x in xs])
        vmin = -vmax
    else:
        vmin = -crange
        vmax = crange
    ims = []
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('RdBu')
    if n_ims == 1:
        # ffs matplotlib "helps" by flattening the array. single subplot is not a thing.
        iter_axs = [axs]
    else:
        iter_axs = axs

    for imi, (x, name, ax) in enumerate(zip(xs, names, iter_axs)):
        ax.set_axis_off()
        ax.set_title(name)
        im = ax.imshow(x, interpolation='bilinear', norm=norm, cmap=cmap)
        ims.append(im)

    fig.colorbar(ims[0], ax=axs, orientation='vertical', shrink = 0.6)
    # fig.tight_layout()
    return fig


def pred_actual_plot(pred, y, * args, **kwargs):
    return multi_heatmap([pred, y, pred-y], names=['pred', 'actual', 'diff'], *args, **kwargs)


def meshify(X,Z):
    """
    takes a 2D array X, containing pairs of grid coordinates, and an array Z containing values at those coordinates, and return a meshgrid-compatible representation of X, and corresponding Z values
    """
    X = np.asarray(X)
    Z = np.asarray(Z)
    sorter = np.lexsort((X[:,1], X[:,0]))
    X = X[sorter]
    Z = Z[sorter]
    n_x = np.unique(X[:,0]).shape[0]
    n_y = np.unique(X[:,1]).shape[0]

    # mesh_X = rearrange(
    #     X, "n (x y) -> x y n", x=n_x, y=n_y
    # )
    mesh_x = X[:, 0].reshape(n_x, n_y)
    mesh_y = X[:, 1].reshape(n_x, n_y)
    mesh_Z = Z.reshape(n_x, n_y)
    return mesh_x, mesh_y, mesh_Z

