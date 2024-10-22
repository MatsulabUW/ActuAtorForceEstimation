# Copyright (c) 2022 Eleanor Jung, Cuncheng Zhu, and Christopher T. lee
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from pathlib import Path

import automembrane.plot_helper as ph
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from PIL import Image
from tqdm.auto import tqdm

from typing import Union
from actuator_constants import image_microns_per_pixel, raw_image_paths

from automembrane.geometry import ClosedPlaneCurveGeometry

ph.matplotlibStyle(small=10, medium=12, large=14)

# Plotting settings
padding = 100
cm = mpl.cm.viridis_r


def make_movie(
    c: npt.NDArray[np.float64],
    e: npt.NDArray[np.float64],
    f: npt.NDArray[np.float64],
    original_coords: npt.NDArray[np.float64],
    file: Path,
    fps: int = 30,
    dpi: int = 100,
    skip: int = 100,
    interactive: bool = False,
    base_path: Union[str, Path] = "movies",
) -> None:
    """Make a movie given a trajectory

    Args:
        c (npt.NDArray[np.float64]): Trajectory coordinates
        e (npt.NDArray[np.float64]): Trajectory of energies
        f (npt.NDArray[np.float64]): Trajectory of forces
        original_coords (npt.NDArray[np.float64]): Original coordinates
        file (Path): File to consider
        fps (int, optional): Frames per second to render. Defaults to 30.
        dpi (int, optional): Dots per inch of raster image. Defaults to 100.
        skip (int, optional): Frame output frequency. Defaults to 100.
        interactive (bool, optional): Whether to display plot interactively. Defaults to False.
    """

    n_iter = c.shape[0]

    x_lim = np.array([np.min(original_coords[:, 0]), np.max(original_coords[:, 0])]) + [
        -padding,
        padding*1.5,
    ]
    y_lim = np.array([np.min(original_coords[:, 1]), np.max(original_coords[:, 1])]) + [
        -padding,
        padding*2.3,
    ]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(autoscale_on=False, xlim=x_lim, ylim=y_lim)

    # flip y-axis
    ax.set_ylim(ax.get_ylim()[::-1])

    (original_line,) = ax.plot(original_coords[:, 0], original_coords[:, 1], color="k")

    (line,) = ax.plot(c[0][:, 0], c[0][:, 1], color="r")
    forces = np.sum(f[0], axis=0)
    f_mag = np.linalg.norm(forces, axis=1)

    q = ax.quiver(
        c[0][:, 0],
        c[0][:, 1],
        forces[:, 0],
        forces[:, 1],
        f_mag,
        cmap=mpl.cm.viridis_r,
        angles="xy",
        units="xy",
        label="force",
        scale=4e1,
        scale_units="xy",
        width=0.1,
        zorder=10,
    )
    ax.set_ylabel(r"Y (nm)")
    ax.set_xlabel(r"X (nm)")

    time_template = "Iteration = %d"
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    with Image.open(raw_image_paths[file.stem]) as im:
        pixel_scale = image_microns_per_pixel[file.stem]
        x_lim_pix = (x_lim / pixel_scale).round()
        y_lim_pix = (y_lim / pixel_scale).round()

        im = im.crop((x_lim_pix[0], y_lim_pix[0], x_lim_pix[1], y_lim_pix[1]))

        plt.imshow(
            im,
            alpha=0.6,
            extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
            zorder=0,
            cmap=plt.cm.Greys_r,
        )

    # Shrink current axis
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    cbar = fig.colorbar(
        q,
        ax=ax,
    )
    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.set_ylabel(r"Force Density ($\mathregular{pN/\mu m^2}$)", rotation=270)


    def animate(i):
        line.set_data(c[i][:, 0], c[i][:, 1])
        time_text.set_text(time_template % (i))

        forces = np.sum(f[i], axis=0)
        f_mag = np.linalg.norm(forces, axis=1)
        cmap = mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=f_mag.min(), vmax=f_mag.max()), cmap=cm
        )
        cbar.update_normal(cmap)
        q.set_UVC(forces[:, 0], forces[:, 1], f_mag)
        q.set_offsets(c[i])
        q.set_clim(f_mag.min(), f_mag.max())
        return line, time_text, q, cbar

    if interactive:
        ## INTERACTIVE ANIMATION
        ani = animation.FuncAnimation(
            fig, animate, np.arange(0, n_iter, 100), interval=200, blit=True
        )
        plt.show()
    else:
        ## MAKE MOVIE
        moviewriter = animation.FFMpegWriter(fps=fps)
        with moviewriter.saving(
            fig, f"{base_path}/{file.stem}_relaxation.mp4", dpi=dpi
        ):
            for i in tqdm(np.arange(0, n_iter + 1, skip)):
                animate(i)
                moviewriter.grab_frame()
    ax.clear()
    plt.close(fig)


def make_nondimensional_movie(
    file_stem: str,
    original_coords: npt.NDArray[np.float64],
    relaxed_coords: npt.NDArray[np.float64],
    relaxed_force: dict[float, npt.NDArray[np.float64]],
    Ksg_range: npt.NDArray[np.float64],
    x_lim: npt.NDArray[np.float64] = None,
    y_lim: npt.NDArray[np.float64] = None,
    style="quiver",
):


    x_shift = -5
    y_shift = -10
    original_coords[:, 0] += x_shift
    original_coords[:, 1] += y_shift
    relaxed_coords[:, 0] += x_shift
    relaxed_coords[:, 1] += y_shift


    fig = plt.figure(figsize=(5, 5), dpi=300)
    spec = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[1, 40])

    # bending-tension parameter bar
    ax1 = fig.add_subplot(
        spec[0],
        autoscale_on=False,
        xlim=(np.min(Ksg_range), np.max(Ksg_range)),
        ylim=(0, 1),
    )
    physics_marker = ax1.axvline(
        Ksg_range[0], 0, 1, linestyle="solid", color="r", linewidth=6
    )

    ax1.set_xticks(
        [
            np.min(Ksg_range),
            (np.min(Ksg_range) + np.max(Ksg_range)) / 2,
            np.max(Ksg_range),
        ]
    )
    ax1.set_xticklabels(["Bending", "Transition", "Tension"])
    ax1.get_yaxis().set_visible(False)
    ax1.xaxis.tick_top()

    # nucleus cell trace
    if x_lim is None:
        x_lim = np.array(
            [np.min(relaxed_coords[:, 0]), np.max(relaxed_coords[:, 0])]
        ) + [
            -padding/1.2,
            padding,
        ]
    if y_lim is None:
        y_lim = np.array(
            [np.min(relaxed_coords[:, 1]), np.max(relaxed_coords[:, 1])]
        ) + [
            -padding/2.3,
            padding*1.2,
        ]

    # if file_stem == "34D-grid2-s3-acta1_001_16":

    ax2 = fig.add_subplot(spec[1], autoscale_on=False, xlim=x_lim, ylim=y_lim)
    # flip y-axis
    ax2.set_ylim(ax2.get_ylim()[::-1])

    with Image.open(raw_image_paths[file_stem]) as im:
        # pixel_scale = image_microns_per_pixel[file_stem]
        # x_lim_pix = (x_lim / pixel_scale).round()
        # y_lim_pix = (y_lim / pixel_scale).round()
        # im = im.crop((x_lim_pix[0], y_lim_pix[0], x_lim_pix[1], y_lim_pix[1]))
        im = im.rotate(8, expand=False)

        ax2.imshow(
            im,
            alpha=0.6,
            extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
            zorder=0,
            cmap=plt.cm.Greys_r,
            aspect='auto',
        )        

    ax2.set_ylabel(r"Y (nm)")
    ax2.set_xlabel(r"X (nm)")

    # Shrink current axis
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    forces = np.sum(relaxed_force[Ksg_range[0]], axis=0)

    max_norm = np.max(np.linalg.norm(forces, axis=1))
    normalized_relaxed_force = forces / max_norm
    f_mag = np.linalg.norm(normalized_relaxed_force, axis=1)

    if style == "quiver":
        q = ax2.quiver(
            relaxed_coords[:, 0],
            relaxed_coords[:, 1],
            -normalized_relaxed_force[:, 0],
            -normalized_relaxed_force[:, 1],
            f_mag,
            # cmap=mpl.cm.viridis,
            cmap=mpl.cm.viridis_r,
            angles="xy",
            # units="xy",
            label="force",
            scale=0.02,
            scale_units="xy",
            width=0.005,
            # zorder=10,
        )
        cbar = fig.colorbar(
            q,
            ax=ax2,
        )
        cbar.ax.get_yaxis().labelpad = 20
        cbar.ax.set_ylabel("Actin Force Density (1)", rotation=270)
    elif style == "color":
        vertex_normal = ClosedPlaneCurveGeometry.vertex_normal(relaxed_coords)
        signed_f_mag = np.sum(forces * vertex_normal, axis=1)
        signed_f_mag = signed_f_mag / np.max(abs(signed_f_mag))
        points = relaxed_coords.reshape(-1, 1, 2)
        print(points.shape)

        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        from matplotlib.collections import LineCollection

        norm = plt.Normalize(-abs(signed_f_mag).max(), abs(signed_f_mag).max())
        lc = LineCollection(segments, cmap="PRGn", norm=norm)
        lc.set_array(signed_f_mag)
        lc.set_linewidth(4)
        q = ax2.add_collection(lc)
        cbar = fig.colorbar(q, ax=ax2, ticks=[-1, 0, 1])
        cbar.ax.set_yticklabels(["Pull", "0", "Push"])
        # curvature_scale = np.max(ClosedPlaneCurveGeometry.edge_curvature(relaxed_coords))
        cbar.ax.get_yaxis().labelpad = 20

    (line,) = ax2.plot(
        relaxed_coords[:, 0], 
        relaxed_coords[:, 1], 
        # "--",
        "o",
        # linewidth=0.2, 
        linewidth=2, 
        color="r"
    )

    # outline plots
    (original_line,) = ax2.plot(
        original_coords[:, 0],
        original_coords[:, 1],
        "o",
        markersize=0.5,
        # linewidth=0.1,
        color="k",
    )

    def animate(Ksg_):
        physics_marker.set_xdata(np.array([Ksg_, Ksg_]))
        # physics_marker.set_xdata(np.array(Ksg_, Ksg_))

        forces = np.sum(relaxed_force[Ksg_], axis=0)
        if style == "quiver":
            max_norm = np.max(np.linalg.norm(forces, axis=1))
            normalized_force = forces / max_norm
            f_mag = np.linalg.norm(normalized_force, axis=1)
            q.set_UVC(-normalized_force[:, 0], -normalized_force[:, 1], f_mag)
            return physics_marker, q
        elif style == "color":
            signed_f_mag = np.sum(forces * vertex_normal, axis=1)
            signed_f_mag = signed_f_mag / np.max(abs(signed_f_mag))
            lc.set_array(signed_f_mag)
            return physics_marker, lc
        return physics_marker

    moviewriter = animation.FFMpegWriter(bitrate=2.5e4, fps=30)

    with moviewriter.saving(fig, f"movies/{file_stem}_nondim_variation.mp4", dpi=300):
        for Ksg_ in tqdm(
            np.concatenate((Ksg_range, np.flip(Ksg_range))), desc="Rendering frames"
        ):
            plt.savefig(f"figures/{file_stem}_Ksg{np.floor(Ksg_*1000)}" + ".png")
            animate(Ksg_)
            moviewriter.grab_frame()            


    # for Ksg_ in tqdm(
    #     np.concatenate((Ksg_range, np.flip(Ksg_range))), desc="Rendering frames"
    # ):
    #     plt.savefig(f"figures/{file_stem}_Ksg{np.floor(Ksg_*1000)}" + ".png")


    ax1.clear()
    ax2.clear()
    plt.close(fig)




def plot_dimensional(
    file_stem: str,
    original_coords: npt.NDArray[np.float64],
    relaxed_coords: npt.NDArray[np.float64],
    forces: npt.NDArray[np.float64],
):

    x_shift = -5
    y_shift = -10
    original_coords[:, 0] += x_shift
    original_coords[:, 1] += y_shift
    relaxed_coords[:, 0] += x_shift
    relaxed_coords[:, 1] += y_shift


    fig = plt.figure(figsize=(5, 5), dpi=300)
    spec = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[1, 40])

    x_lim = np.array(
        [np.min(relaxed_coords[:, 0]), np.max(relaxed_coords[:, 0])]
    ) + [
        -padding/1.2,
        padding,
    ]
    y_lim = np.array(
        [np.min(relaxed_coords[:, 1]), np.max(relaxed_coords[:, 1])]
    ) + [
        -padding/2.3,
        padding*1.2,
    ]

    ax2 = fig.add_subplot(autoscale_on=False, xlim=x_lim, ylim=y_lim)
    # flip y-axis
    ax2.set_ylim(ax2.get_ylim()[::-1])

    with Image.open(raw_image_paths[file_stem]) as im:
        im = im.rotate(8, expand=False)
        ax2.imshow(
            im,
            alpha=0.6,
            extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
            zorder=0,
            cmap=plt.cm.Greys_r,
            aspect='auto',
        )        

    ax2.set_ylabel(r"Y (nm)")
    ax2.set_xlabel(r"X (nm)")

    # Shrink current axis
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # max_norm = np.max(np.linalg.norm(forces, axis=1))
    # normalized_relaxed_force = forces / max_norm
    f_mag = np.linalg.norm(forces, axis=1)

    q = ax2.quiver(
        relaxed_coords[:, 0],
        relaxed_coords[:, 1],
        -forces[:, 0],
        -forces[:, 1],
        f_mag,
        # cmap=mpl.cm.viridis,
        cmap=mpl.cm.viridis_r,
        angles="xy",
        # units="xy",
        label="force",
        scale=0.003,
        # scale=0.01,
        scale_units="xy",
        width=0.005,
        # zorder=10,
    )
    cbar = fig.colorbar(
        q,
        ax=ax2,
    )
    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.set_ylabel(r"Force Density $\mathrm{(pN/nm^2)}$", rotation=270)

    (line,) = ax2.plot(
        relaxed_coords[:, 0], 
        relaxed_coords[:, 1], 
        # "--",
        "o",
        linewidth=2, 
        color="r"
    )

    # outline plots
    (original_line,) = ax2.plot(
        original_coords[:, 0],
        original_coords[:, 1],
        "o",
        markersize=0.5,
        color="k",
    )

    return fig




# if __name__ == "__main__":
#     ## BATCH RENDER
#     f_movie = partial(make_movie, fps=30, dpi=200, skip=100)
#     r = process_map(f_movie, files, max_workers=6)
