# Copyright (c) 2022 Eleanor Jung, Cuncheng Zhu, and Christopher T. Lee
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Union

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from .energy import Material, ClosedPlaneCurveMaterial


def fwd_euler_integrator(
    coords: npt.NDArray[np.float64],
    mat: Material,
    n_steps: int = int(1e5),
    dt: float = 5e-6,
    save_trajectory: bool = False,
) -> Union[
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
]:
    """Perform forward Euler integration

    Args:
        coords (npt.NDArray[np.float64]): Vertex coordinate position
        mat (Material): Membrane material holding parameters
        n_steps (int, optional): Number of steps to take. Defaults to int(1e5).
        dt (float, optional): Time step. Defaults to 5e-6.
        save_trajectory (bool, optional): Whether or not to save forces. Defaults to False.

    Returns:
       Union[ tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]], ]: Returns final coordinates and energies or trajectory of coordinates, energies and forces.
    """

    is_closed_curve = isinstance(mat, ClosedPlaneCurveMaterial)
    # energy_shape = mat._energy(coords).shape
    energy_shape = mat.get_energy_shape(coords)
    # energy_shape = 3

    # energy_log = np.zeros((n_steps + 1, 3))
    energy_log = np.zeros((n_steps + 1, *energy_shape))
    if save_trajectory:
        coord_log = np.zeros((n_steps + 1, *coords.shape))
        force_log = np.zeros(
            (
                n_steps + 1,
                *energy_shape,
                *coords.shape,
            )
        )

        coord_log[0] = coords

        for i in tqdm(range(0, n_steps + 1), desc="Energy relaxation"):
            # Compute dual length
            delta_coords = np.roll(coords[:-1], -1, axis=0) - coords[:-1]
            edgeLengths = np.linalg.norm(delta_coords, axis=1)
            dualLengths = ((edgeLengths + np.roll(edgeLengths, 1)) / 2.0).reshape(-1, 1)
            dualLengths = np.vstack((dualLengths, dualLengths[0]))

            # Calculate energy and force
            energy_log[i], force = mat._energy_force(coords)
            force_log[i] = force / dualLengths

            # c_t+1 = c_t - force * dt
            coords = np.array(coords + np.sum(force, axis=0) * dt)
            if is_closed_curve:
                coords[-1] = coords[0]
            if i < n_steps:
                coord_log[i + 1] = coords
        return coord_log, energy_log, force_log

    else:
        for i in tqdm(range(0, n_steps), desc="Energy relaxation"):
            energy_log[i], force = mat._energy_force(coords)
            
            # c_t+1 = c_t + force * dt
            coords = np.array(coords + np.sum(force, axis=0) * dt)
            # coords[1:-1] = np.array(coords[1:-1] + np.sum(force, axis=0)[1:-1] * dt) # pinned
            # coords[3:-3] = np.array(coords[3:-3] + np.sum(force, axis=0)[3:-3] * dt) # fixed

            # print(coords[2])

            if is_closed_curve:
                coords[-1] = coords[0]
        if is_closed_curve:
            energy_log[-1] = mat._energy(coords)
        return coords, energy_log



# Original
def fwd_euler_integrator_(
    coords: npt.NDArray[np.float64],
    mat: Material,
    n_steps: int = int(1e5),
    dt: float = 5e-6,
    save_trajectory: bool = False,
) -> Union[
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
]:
    """Perform forward Euler integration

    Args:
        coords (npt.NDArray[np.float64]): Vertex coordinate position
        mat (Material): Membrane material holding parameters
        n_steps (int, optional): Number of steps to take. Defaults to int(1e5).
        dt (float, optional): Time step. Defaults to 5e-6.
        save_trajectory (bool, optional): Whether or not to save forces. Defaults to False.

    Returns:
       Union[ tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]], ]: Returns final coordinates and energies or trajectory of coordinates, energies and forces.
    """

    energy_shape = mat._energy(coords).shape

    energy_log = np.zeros((n_steps + 1, *energy_shape))
    if save_trajectory:
        coord_log = np.zeros((n_steps + 1, *coords.shape))
        force_log = np.zeros(
            (
                n_steps + 1,
                *energy_shape,
                *coords.shape,
            )
        )

        coord_log[0] = coords

        for i in tqdm(range(0, n_steps + 1), desc="Energy relaxation"):
            # Compute dual length
            delta_coords = np.roll(coords[:-1], -1, axis=0) - coords[:-1]
            edgeLengths = np.linalg.norm(delta_coords, axis=1)
            dualLengths = ((edgeLengths + np.roll(edgeLengths, 1)) / 2.0).reshape(-1, 1)
            dualLengths = np.vstack((dualLengths, dualLengths[0]))

            # Calculate energy and force
            energy_log[i], force = mat._energy_force(coords)
            force_log[i] = force / dualLengths

            # c_t+1 = c_t - force * dt
            coords = np.array(coords + np.sum(force, axis=0) * dt)
            coords[-1] = coords[0]
            if i < n_steps:
                coord_log[i + 1] = coords
        return coord_log, energy_log, force_log

    else:
        for i in tqdm(range(0, n_steps), desc="Energy relaxation"):
            energy_log[i], force = mat._energy_force(coords)
            # c_t+1 = c_t + force * dt
            coords = np.array(coords + np.sum(force, axis=0) * dt)
            coords[-1] = coords[0]
        energy_log[-1] = mat._energy(coords)
        return coords, energy_log
