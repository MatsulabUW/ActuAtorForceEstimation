import numpy as np
from automembrane.energy import ClosedPlaneCurveMaterial
from automembrane.energy import OpenPlaneCurveMaterial
from automembrane.geometry import OpenPlaneCurveGeometry


def get_force_density(parameters, coords, spont_c = None):

    # mem = ClosedPlaneCurveMaterial(**parameters)
    mem = OpenPlaneCurveMaterial(**parameters)
    forces = np.array(
        [
            # force / ClosedPlaneCurveGeometry.vertex_dual_length(coords)
            force / OpenPlaneCurveGeometry.vertex_dual_length(coords)
            for force in mem.force(coords, spont_c)
            # for force in mem.force(coords, OpenPlaneCurveGeometry.edge_curvature(coords))
        ]
    )
    return forces
