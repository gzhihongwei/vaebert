import os

import numpy as np
import pandas as pd
import seaborn as sns
import torch

import matplotlib.pyplot as plt
import skimage.measure as sm
import plotly.figure_factory as FF

from plotly.offline import plot
from sklearn.manifold import TSNE


def tsne(latents: np.ndarray, data: pd.DataFrame) -> None:
    tsne = TSNE(
        n_components=2, n_iter=1100, verbose=3, perplexity=40, learning_rate=200
    )
    lvects = tsne.fit_transform(latents)
    data["tsne1"] = lvects[:, 0]
    data["tsne2"] = lvects[:, 1]
    plt.rcParams["figure.figsize"] = (12, 8)
    sns.set_style("darkgrid")
    ax = sns.scatterplot(
        data=data,
        x="tsne1",
        y="tsne2",
        hue="Category",
        s=10,
        linewidth=0,
        palette="bright",
    )
    # Clear x axis stuff
    ax.set(xticklabels=[])
    ax.set(xlabel=None)
    ax.tick_params(bottom=False)
    # Clear y axis stuff
    ax.set(yticklabels=[])
    ax.set(ylabel=None)
    ax.tick_params(left=False)
    plt.show()


# https://github.com/starstorms9/shape/blob/master/utils.py#L380
def plotMesh(verts, faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        verts[:, 0], verts[:, 1], faces, verts[:, 2], linewidth=0.2, antialiased=True
    )
    plt.show()


def plotVox(
    voxin,
    step=1,
    title="",
    outfile_path=".",
    threshold=0.5,
    limits=None,
    show_axes=True,
    save_fig=False,
    show_fig=True,
):
    vox = np.squeeze(voxin)

    vflat = vox.flatten()
    if threshold is None:
        threshold = (np.min(vflat) + np.max(vflat)) / 2

    if not np.any(vox):
        print("No voxels for: {}".format(title))
        vflat = voxin.flatten()
        plt.hist(vflat, bins=10)
        plt.suptitle(title)
        return

    try:
        verts, faces = createMesh(vox, step, threshold)
    except:
        print("Failed creating mesh for voxels.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if not show_axes:
        ax.axis("off")

    if limits is not None:
        ax.set_xlim(0, limits[0])
        ax.set_ylim(0, limits[1])
        ax.set_zlim(0, limits[2])

    _ = ax.plot_trisurf(
        verts[:, 0], verts[:, 1], faces, verts[:, 2], linewidth=0.2, antialiased=True
    )
    plt.suptitle(title)

    if save_fig:
        _ = fig.savefig(os.path.join(outfile_path, title))
    if show_fig:
        _ = plt.show()
    return


def createMesh(vox, step=1, threshold=0.5):
    vox = np.pad(vox, step)
    verts, faces, *_ = sm.marching_cubes(vox, threshold, step_size=step)
    return verts, faces


def showMesh(verts, faces, aspect=None, plot_it=True):
    if aspect is None:
        aspect = dict(x=1, y=1, z=1)
    fig = FF.create_trisurf(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        simplices=faces,
        title="Mesh",
        aspectratio=aspect,
    )
    if plot_it:
        plot(fig)
    return fig


def showReconstruct(
    model,
    samples,
    index=2,
    title="",
    show_original=True,
    show_reconstruct=True,
    save_fig=False,
    limits=None,
):
    xvox = torch.tensor(samples[index]).unsqueeze(0).float()
    predictions = model(xvox).detach().numpy()

    if np.max(predictions) < 0.5:
        print("No voxels")
        return

    if show_original:
        plotVox(
            xvox,
            step=1,
            title="Original {}".format(title),
            threshold=0.5,
            stats=False,
            save_fig=save_fig,
            limits=limits,
        )
    if show_reconstruct:
        plotVox(
            predictions,
            step=1,
            title="Reconstruct {}".format(title),
            threshold=0.5,
            stats=False,
            save_fig=save_fig,
            limits=limits,
        )
