# Copyright (c) 2022 Eleanor Jung, Cuncheng Zhu, and Christopher T. Lee
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt


class ClosedPlaneCurveGeometry:
    @staticmethod
    @jax.jit
    def edge_curvature(
        vertex_positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute edge curvature
        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates

        Returns:
            npt.NDArray[np.float64]: edge curvature
        """
        d_pos = jnp.roll(vertex_positions[:-1], -1, axis=0) - vertex_positions[:-1]
        edgeLengths = jnp.linalg.norm(d_pos, axis=1)

        edgeAbsoluteAngles = jnp.arctan2(d_pos[:, 1], d_pos[:, 0])
        vertexTurningAngles = (
            jnp.roll(edgeAbsoluteAngles, -1) - edgeAbsoluteAngles
        ) % (2 * jnp.pi)
        vertexTurningAngles = (vertexTurningAngles + jnp.pi) % (2 * jnp.pi) - jnp.pi

        tan_vertex_turning_angles = jnp.tan(vertexTurningAngles / 2)
        edgeCurvatures = (
            tan_vertex_turning_angles + jnp.roll(tan_vertex_turning_angles, 1)
        ) / edgeLengths
        return edgeCurvatures

    @staticmethod
    @jax.jit
    def vertex_dual_length(
        vertex_positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute dual edge length
        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates

        Returns:
            npt.NDArray[np.float64]: vertex dual length
        """
        dc = jnp.roll(vertex_positions[:-1], -1, axis=0) - vertex_positions[:-1]
        edgeLengths = jnp.linalg.norm(dc, axis=1)
        dualLengths = ((edgeLengths + jnp.roll(edgeLengths, 1)) / 2.0).reshape(-1, 1)
        return jnp.vstack((dualLengths, dualLengths[0]))

    @staticmethod
    @jax.jit
    def edge_length(
        vertex_positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute edge length
        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates

        Returns:
            npt.NDArray[np.float64]: edge length
        """
        dc = jnp.roll(vertex_positions[:-1], -1, axis=0) - vertex_positions[:-1]
        return jnp.linalg.norm(dc, axis=1)

    @staticmethod
    def vertex_normal(
        vertex_positions: npt.NDArray[np.float64],
        orientation: str = "cw",
    ) -> npt.NDArray[np.float64]:
        """Compute length weighted vertex normal
        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates
            orientation (str): normal orientation convention

        Returns:
            npt.NDArray[np.float64]: vertex normal
        """
        dc = np.roll(vertex_positions[:-1], -1, axis=0) - vertex_positions[:-1]
        if orientation == "ccw":
            edge_normal = np.stack([-dc[:, 1], dc[:, 0]], axis=1)
        elif orientation == "cw":
            edge_normal = np.stack([dc[:, 1], -dc[:, 0]], axis=1)
        else:
            raise RuntimeError("Orientation is either 'ccw' or 'cw'!")

        vertex_normal = edge_normal + np.roll(edge_normal, 1, axis=0)
        vertex_normal = vertex_normal / np.linalg.norm(vertex_normal, axis=1).reshape(
            -1, 1
        )
        vertex_normal = np.vstack((vertex_normal, vertex_normal[0]))
        return vertex_normal



class OpenPlaneCurveGeometry:
    @staticmethod
    @jax.jit
    def vertex_dual_length(
        vertex_positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute dual edge length
        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates

        Returns:
            npt.NDArray[np.float64]: vertex dual length
        """
        dc = jnp.diff(vertex_positions, axis=0)
        edgeLengths = jnp.linalg.norm(dc, axis=1)
        edgeLengths = jnp.append(edgeLengths, edgeLengths[-1])
        edgeLengths = jnp.append(edgeLengths[0], edgeLengths)
        dualLengths = ((edgeLengths[:-1] + edgeLengths[1:]) / 2.0).reshape(-1, 1)

        return dualLengths

    @staticmethod
    @jax.jit
    def edge_curvature(
        vertex_positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute edge curvature
        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates

        Returns:
            npt.NDArray[np.float64]: edge curvature
        """

        d_pos = jnp.diff(vertex_positions, axis=0)
        edgeLengths = jnp.linalg.norm(d_pos, axis=1)

        edgeAbsoluteAngles = jnp.arctan2(d_pos[:, 1], d_pos[:, 0])
        vertexTurningAngles = (jnp.diff(edgeAbsoluteAngles)) % (2 * jnp.pi)
        vertexTurningAngles = (vertexTurningAngles + jnp.pi) % (2 * jnp.pi) - jnp.pi
        vertexTurningAngles = jnp.append(vertexTurningAngles, vertexTurningAngles[-1])
        vertexTurningAngles = jnp.append(vertexTurningAngles[0], vertexTurningAngles)

        tan_vertex_turning_angles = jnp.tan(vertexTurningAngles / 2)

        edgeCurvatures = (
            tan_vertex_turning_angles[:-1] + tan_vertex_turning_angles[1:]
        ) / edgeLengths

        return edgeCurvatures

    @staticmethod
    @jax.jit
    def edge_length(
        vertex_positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute edge length
        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates

        Returns:
            npt.NDArray[np.float64]: edge length
        """
        dc = jnp.diff(vertex_positions, axis=0)
        return jnp.linalg.norm(dc, axis=1)


    @staticmethod
    def vertex_normal(
        vertex_positions: npt.NDArray[np.float64],
        orientation: str = "cw",
    ) -> npt.NDArray[np.float64]:
        """Compute length weighted vertex normal
        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates
            orientation (str): normal orientation convention

        Returns:
            npt.NDArray[np.float64]: vertex normal
        """
        dc = jnp.diff(vertex_positions, axis=0)
        if orientation == "ccw":
            edge_normal = np.stack([-dc[:, 1], dc[:, 0]], axis=1)
        elif orientation == "cw":
            edge_normal = np.stack([dc[:, 1], -dc[:, 0]], axis=1)
        else:
            raise RuntimeError("Orientation is either 'ccw' or 'cw'!")


        edge_normal = jnp.vstack((edge_normal, edge_normal[-1, :]))  # Add last edge normal
        edge_normal = jnp.vstack((edge_normal[0, :], edge_normal))   # Add first edge normal


        vertex_normal = (edge_normal[:-1] + edge_normal[1:])
        vertex_normal /= np.linalg.norm(vertex_normal, axis=1).reshape(-1, 1)

        return vertex_normal


