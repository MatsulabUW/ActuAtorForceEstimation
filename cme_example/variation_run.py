# Copyright (c) 2022 Eleanor Jung, Cuncheng Zhu, and Christopher T. lee
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from functools import partial

import automembrane.plot_helper as ph
import jax
import numpy as np
from automembrane.energy import ClosedPlaneCurveMaterial
from automembrane.energy import OpenPlaneCurveMaterial
from automembrane.geometry import OpenPlaneCurveGeometry
from tqdm.contrib.concurrent import process_map

from actuator_constants import files

jax.config.update("jax_enable_x64", True)
ph.matplotlibStyle(small=10, medium=12, large=14)


def get_dimensional_tension(Ksg_, Kb, coords):
    # curvature_scale = float(np.max(np.abs(ClosedPlaneCurveGeometry.edge_curvature(coords))))
    curvature_scale = float(np.max(np.abs(OpenPlaneCurveGeometry.edge_curvature(coords))))
    # print(curvature_scale)
    return 4 * Kb * curvature_scale**2 * Ksg_

def get_average_edge_length(coords):
    return float(np.mean(OpenPlaneCurveGeometry.edge_length(coords)))


# def get_force_density(parameters, coords):
#     # mem = ClosedPlaneCurveMaterial(**parameters)
#     mem = OpenPlaneCurveMaterial(**parameters)
#     forces = np.array(
#         [
#             # force / ClosedPlaneCurveGeometry.vertex_dual_length(coords)
#             force / OpenPlaneCurveGeometry.vertex_dual_length(coords)
#             # force
#             for force in mem.force(coords)
#         ]
#     )
#     return forces

def get_force_density(parameters, coords, spont_c = None):

    # mem = ClosedPlaneCurveMaterial(**parameters)
    # mem = OpenPlaneCurveMaterial(**parameters)
    mem = OpenPlaneCurveMaterial(**parameters, spont_curvatures = spont_c)
    forces = np.array(
        [
            # force / ClosedPlaneCurveGeometry.vertex_dual_length(coords)
            force / OpenPlaneCurveGeometry.vertex_dual_length(coords)
            for force in mem.force(coords)
            # for force in mem.force(coords, spont_c)
            # for force in mem.force(coords, OpenPlaneCurveGeometry.edge_curvature(coords))
        ]
    )
    return forces

def get_force(parameters, coords, spont_c = None):

    # mem = ClosedPlaneCurveMaterial(**parameters)
    # mem = OpenPlaneCurveMaterial(**parameters)
    mem = OpenPlaneCurveMaterial(**parameters, spont_curvatures = spont_c)
    forces = mem.force(coords)
    return forces


def run_parameter_variation(file, _Ksg_):
    Ksg_force = {}
    data = np.load(f"relaxed_coords/{file.stem}.npz")
    relaxed_coords = data["relaxed_coords"]
    for Ksg_ in _Ksg_:
        # Instantiate material properties
        Kb = 0.1
        Ksg = get_dimensional_tension(Ksg_, Kb, relaxed_coords)
        parameters = {
            "Kb": Kb / 4,  # Bending modulus (pN um; original 1e-19 J)
            "Ksg": Ksg,  # Global stretching modulus (pN um/um^2; original 0.05 mN/m)
            "Ksl": 0,
        }
        print(f"dimensional tension for {file.stem}: ", Ksg)
        # Ksg_coords_force.append(np.concatenate(([relaxed_coords], relaxed_forces), axis=0))
        Ksg_force[Ksg_] = get_force_density(parameters, relaxed_coords)
    Ksg_force = np.array([Ksg_force])
    np.savez(f"forces/{file.stem}", Ksg_range=_Ksg_, Ksg_force=Ksg_force)


if __name__ == "__main__":
    f_run = partial(run_parameter_variation, _Ksg_=np.linspace(0, 0.3, 1 + 2**5))
    r = process_map(f_run, files, max_workers=12)
        