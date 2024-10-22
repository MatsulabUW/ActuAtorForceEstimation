# Copyright (c) 2022 Eleanor Jung, Cuncheng Zhu, and Christopher T. lee
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from functools import partial
from pathlib import Path
from typing import Union

import automembrane.plot_helper as ph
import jax
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from automembrane.energy import ClosedPlaneCurveMaterial
from automembrane.energy import OpenPlaneCurveMaterial
from automembrane.integrator import fwd_euler_integrator
from PIL import Image
from scipy.interpolate import splev, splprep
from tqdm.contrib.concurrent import process_map


from actuator_constants import image_microns_per_pixel, raw_image_paths, files

from make_movie import make_movie

jax.config.update("jax_enable_x64", True)
ph.matplotlibStyle(small=10, medium=12, large=14)


def resample(
    original_coords: npt.NDArray[np.float64], n_vertices: int = 1000
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Resample discrete plane curve into n_vertices

    Args:
        original_coords (npt.NDArray[np.float64]): Original coordinates
        n_vertices (int, optional): Number of vertices to resample to. Defaults to 1000.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: New coordinates and B-spline parameters
    """
    # total_length = np.sum(
    #     np.linalg.norm(
    #         np.roll(original_coords[:-1], -1, axis=0) - original_coords[:-1], axis=1
    #     )
    # )
    # n_vertices = math.floor(total_length / target_edge_length)
    # print(f"  Resampling to {n_vertices} vertices")

    # Cubic B-spline interpolation with no smoothing (s=0)
    # tck, _ = splprep([original_coords[:, 0], original_coords[:, 1]], s=0, per=True) #periodic
    tck, _ = splprep([original_coords[:, 0], original_coords[:, 1]], s=0, per=False) #non-periodic

    xi, yi = splev(np.linspace(0, 1, n_vertices), tck)
    coords = np.hstack((xi.reshape(-1, 1), yi.reshape(-1, 1)))
    return coords, tck


def plot_contour(
    fig,
    file_stem: str,
    original_coords: npt.NDArray[np.float64],
    relaxed_coords: npt.NDArray[np.float64],
):
    padding = 100
    x_lim = np.array([np.min(relaxed_coords[:, 0]), np.max(relaxed_coords[:, 0])]) + [
        -padding/1.2,
        padding,
        # padding*1.5,
    ]
    y_lim = np.array([np.min(relaxed_coords[:, 1]), np.max(relaxed_coords[:, 1])]) + [
        -padding/2.3,
        padding*1.2,
        # padding*2.3,
    ]
    ax = fig.add_subplot(autoscale_on=False, xlim=x_lim, ylim=y_lim)

    # flip y-axis
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_ylabel(r"Y (nm)")
    ax.set_xlabel(r"X (nm)")

    # nucleus cell trace
    with Image.open(raw_image_paths[file_stem]) as im:
        # pixel_scale = image_microns_per_pixel[file_stem]
        # x_lim_pix = (x_lim / pixel_scale).round()
        # y_lim_pix = (y_lim / pixel_scale).round()

        # cropped_image = im.crop(
        #     (x_lim_pix[0], y_lim_pix[0], x_lim_pix[1], y_lim_pix[1])
        # )
        rotated_image = im.rotate(8, expand=False)

        im = ax.imshow(
            # cropped_image,
            rotated_image,
            alpha=0.6,
            extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
            zorder=0,
            cmap=plt.cm.Greys_r,
            aspect='auto',
        )

    x_shift = -5
    y_shift = -10
    original_coords[:, 0] += x_shift
    original_coords[:, 1] += y_shift
    relaxed_coords[:, 0] += x_shift
    relaxed_coords[:, 1] += y_shift

    (original_line,) = ax.plot(
        original_coords[:, 0],
        original_coords[:, 1],
        "o",
        markersize=0.5,
        color="k",
        label="Original",
    )
    (line,) = ax.plot(
        relaxed_coords[:, 0],
        relaxed_coords[:, 1],
        # "--",
        "o",
        linewidth=2,
        color="r",
        label="Relaxed",
    )

    ax.legend(loc="upper right", frameon=True, framealpha=0.5)
    return fig, (original_line, line, im)


def plot_contour_frc(
    fig,
    file_stem: str,
    original_coords: npt.NDArray[np.float64],
    relaxed_coords: npt.NDArray[np.float64],
    forces: npt.NDArray[np.float64],
):
    
    forces = np.sum(forces, axis=0)

    # # TEST to remove the boundary element when visualizing it
    # forces = forces[1:]
    # forces = forces[:-1]
    # relaxed_coords = relaxed_coords[1:]
    # relaxed_coords = relaxed_coords[:-1]
    # ###

    padding = 100

    x_lim = np.array([np.min(relaxed_coords[:, 0]), np.max(relaxed_coords[:, 0])]) + [
        -1.1*padding,
        padding*1.5,
    ]
    y_lim = np.array([np.min(relaxed_coords[:, 1]), np.max(relaxed_coords[:, 1])]) + [
        -padding,
        padding*2.3,
    ]
    ax = fig.add_subplot(autoscale_on=False, xlim=x_lim, ylim=y_lim)

    # flip y-axis
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_ylabel(r"Y (nm)")
    ax.set_xlabel(r"X (nm)")


    print(raw_image_paths[file_stem])
    with Image.open(raw_image_paths[file_stem]) as im:
        rotated_im = im.rotate(10, expand=False)
        im = ax.imshow(
            rotated_im,
            alpha=0.6,
            extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
            zorder=0,
            cmap=plt.cm.Greys_r,
            aspect='auto',
        )

    x_shift = -5
    y_shift = -10
    original_coords[:, 0] += x_shift
    original_coords[:, 1] += y_shift
    relaxed_coords[:, 0] += x_shift
    relaxed_coords[:, 1] += y_shift


    (original_line,) = ax.plot(
        original_coords[:, 0],
        original_coords[:, 1],
        "o",
        markersize=2,
        color="k",
        label="Original",
    )
    (line,) = ax.plot(
        relaxed_coords[:, 0],
        relaxed_coords[:, 1],
        # "--",
        "o",
        linewidth=2,
        color="r",
        label="Relaxed",
        alpha=0.7,
    )

    # Plot force vectors using quiver
    force_magnitude = np.sqrt(forces[:, 0]**2 + forces[:, 1]**2)
    avg_force = np.mean(force_magnitude)
    q = ax.quiver(
        relaxed_coords[:, 0],   # X-coordinates of the points
        relaxed_coords[:, 1],   # Y-coordinates of the points
        -forces[:, 0],          # X-component of the force vectors
        -forces[:, 1],          # Y-component of the force vectors
        force_magnitude,        # Color arrows based on the magnitude of forces
        angles="xy",            # Keep the angles consistent with the XY plane
        scale_units="xy",       # Scale the arrows based on the XY plane units
        scale=avg_force * 0.08, # Adjust the scale, play with this value
        width=0.005,            # Adjust the arrow width for better visibility
        cmap="viridis",         # Colormap for arrow colors (choose any cmap)
        alpha=0.9,              # Slightly transparent arrows
    )

    cbar = plt.colorbar(q, ax=ax)
    cbar.set_label(r"Force Density ($\mathregular{pN/nm^2}$)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.5)

    return fig, (original_line, line, im)


def remove_duplicate_points_preserve_order(coords):
    # Use a list to store unique rows while preserving order
    seen = set()
    unique_coords = []
    for coord in coords:
        # Convert row to a tuple to make it hashable for the set
        coord_tuple = tuple(coord)
        if coord_tuple not in seen:
            unique_coords.append(coord)
            seen.add(coord_tuple)
    
    return np.array(unique_coords)


def preprocess_mesh(
    file: Union[str, Path], 
    resample_geometry: bool = True, 
    n_vertices: int = 1000,
    unit_scale: float = 0.6
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Preprocess the plane curve geometry

    Args:
        file (Union[str, Path]): Filename to process
        resample_geometry (bool, optional): Flag to resample geometry. Defaults to True.
        n_vertices (int, optional): number of points to resample to. Defaults to 1000.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: reprocessed and original coordinates
    """

    original_coords = np.loadtxt(file) * unit_scale
    original_coords = remove_duplicate_points_preserve_order(original_coords)
    # original_coords = np.vstack(
    #     (original_coords, original_coords[0])
    # )  # Energy expects last point to equal first

    # mirroring
    # original_coords = original_coords * -1
    # original_coords[:, 0] = original_coords[:, 0] * -1
    original_coords[:, 1] = original_coords[:, 1] * -1

    if resample_geometry:
        coords, _ = resample(original_coords, n_vertices)
    else:
        coords = original_coords
    return coords, original_coords


def relax_bending(coords, Kb, Ksg, Ksl, dt, n_iter, boundary):
    # Instantiate material properties
    parameters = {
        "Kb": Kb / 4,
        "Ksg": Ksg,
        "Ksl": Ksl,
        "boundary": boundary
    }
    # Perform energy relaxation
    if n_iter > 0:
        coords, _ = fwd_euler_integrator(
            coords,
            # ClosedPlaneCurveMaterial(**parameters),
            OpenPlaneCurveMaterial(**parameters),
            n_steps=n_iter,
            dt=dt,
        )
    return coords


relaxation_parameters = {
    "34D-grid2-s2_002_16": [
        {
            "dt": 5e-6,
            "n_iter": int(1e5),
            "Kb": 1,
            "Ksg": 1,
            "Ksl": 0.1,
        }
    ],
    "34D-grid2-s3_028_16": [
        {
            "dt": 1e-6,
            "n_iter": int(1e5),
            "Kb": 1,
            "Ksg": 1,
            "Ksl": 1,
        }
    ],
    "34D-grid2-s5_005_16": [
        {
            "dt": 3e-7,
            "n_iter": int(1e5),
            "Kb": 1,
            "Ksg": 20,
            "Ksl": 1,
        },
        {
            "dt": 3e-7,
            "n_iter": int(5e4),
            "Kb": 1,
            "Ksg": 1,
            "Ksl": 0.1,
        },
    ],
    "34D-grid2-s3-acta1_001_16": [
        {
            "dt": 1e-5,
            "n_iter": int(8e2),
            "Kb": 1,
            "Ksg": 1,
            "Ksl": 0.1,
        }
    ],
    "other": [
        {
            "dt": 1e-5,
            "n_iter": int(1e5),
            "Kb": 1,
            "Ksg": 1,
            "Ksl": 0.1,
        }
    ],
}


def run_relaxation(file: Path, n_vertices: int = 1000):
    coords, original_coords = preprocess_mesh(
        file, resample_geometry=True, n_vertices=n_vertices
    )

    if file.stem in relaxation_parameters:
        for params in relaxation_parameters[file.stem]:
            relaxed_coords = relax_bending(coords, **params)
    else:
        for params in relaxation_parameters["other"]:
            relaxed_coords = relax_bending(coords, **params)

    if file.stem == "34D-grid3-ActA1_007_16":
        relaxed_coords = np.flip(relaxed_coords, axis=0)
        original_coords = np.flip(original_coords, axis=0)

    np.savez(
        f"relaxed_coords/{file.stem}",
        relaxed_coords=relaxed_coords,
        original_coords=original_coords,
    )
    # data = np.load(f"relaxed_coords/{file.stem}.npz")
    # relaxed_coords = data["relaxed_coords"]
    # original_coords = data["original_coords"]

    fig = plt.figure(figsize=(5, 5))
    plot_contour(
        fig,
        file.stem,
        original_coords,
        relaxed_coords,
    )
    fig.set_tight_layout(True)
    plt.savefig("relaxed_coords/" + file.stem + ".png")
    fig.clear()
    plt.close(fig)


def generate_relaxation_movie(file: Path, n_vertices: int = 1000):
    coords, original_coords = preprocess_mesh(
        file, resample_geometry=True, n_vertices=n_vertices
    )

    def get_trajectory(coords, Kb, Ksg, Ksl, dt, n_iter):
        # Instantiate material properties
        parameters = {
            "Kb": Kb / 4,
            "Ksg": Ksg,
            "Ksl": Ksl,
        }
        # Perform energy relaxation
        c, e, f = fwd_euler_integrator(
            coords,
            ClosedPlaneCurveMaterial(**parameters),
            n_steps=n_iter,
            dt=dt,
            save_trajectory=True,
        )
        return c, e, f

    if file.stem in relaxation_parameters:
        c, e, f = get_trajectory(coords, **relaxation_parameters[file.stem][0])
    else:
        c, e, f = get_trajectory(coords, **relaxation_parameters["other"][0])

    make_movie(c, e, f, original_coords, file, skip=1000)
    del c, e, f


if __name__ == "__main__":
    f_run = partial(run_relaxation, n_vertices=1000)
    r = process_map(f_run, files, max_workers=12)

    f_run = partial(generate_relaxation_movie, n_vertices=1000)
    r = process_map(f_run, files, max_workers=1)
