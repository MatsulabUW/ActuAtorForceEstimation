# Copyright (c) 2022 Eleanor Jung, Cuncheng Zhu, and Christopher T. Lee
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


from abc import ABC, abstractmethod
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt



# def point_to_segment_distance(Q, P1, P2):
#     """Calculate the distance from point Q to the line segment P1-P2 with robust handling."""
#     segment_vector = P2 - P1
#     segment_length = jnp.linalg.norm(segment_vector)

#     # Handle the zero-length segment case
#     def zero_length_case():
#         return jnp.linalg.norm(Q - P1)  # Treat the segment as a single point at P1

#     # Normal case
#     def normal_case():
#         t = jnp.dot(Q - P1, segment_vector) / (segment_length ** 2 + 1e-8)  # Avoid division by zero
#         # t = jnp.dot(Q - P1, segment_vector) / (segment_length**2)
#         t = jnp.clip(t, 0.0, 1.0)  # Clamp t to the segment
#         projection = P1 + t * segment_vector
#         return jnp.linalg.norm(Q - projection)

#     # Use conditional logic to handle zero-length segments
#     return jax.lax.cond(segment_length < 1e-8, zero_length_case, normal_case)
#     # return normal_case()


def point_to_segment_distance(Q, P1, P2):
    """Calculate the distance from point Q to the line segment P1-P2 with robust handling."""
    segment_vector = P2 - P1
    segment_length = jnp.linalg.norm(segment_vector)

    # Handle the zero-length segment case
    def zero_length_case():
        return jnp.linalg.norm(Q - P1)

    # Normal case
    def normal_case():
        t = jnp.dot(Q - P1, segment_vector) / (segment_length ** 2 + 1e-8)  # Stabilize denominator
        t = jnp.clip(t, 0.0, 1.0)  # Clamp t to the segment
        projection = P1 + t * segment_vector
        return jnp.linalg.norm(Q - projection)

    # Detect if Q == P1 or Q == P2
    is_at_p1 = jnp.allclose(Q, P1)
    is_at_p2 = jnp.allclose(Q, P2)

    def exact_case():
        return 0.0  # Distance is zero

    # Combine the exact case detection with conditional logic
    return jax.lax.cond(
        is_at_p1 | is_at_p2,  # If Q equals P1 or P2
        exact_case,
        lambda: jax.lax.cond(segment_length < 1e-8, zero_length_case, normal_case),
    )

def distances_to_curve(data_points, vertex_positions):
    """
    Calculate the minimum distance from each data point to a curve.

    Args:
        data_points: np.NDArray[np.float64] or jnp.array, shape (M, D)
            Array of M points in D-dimensional space.
        vertex_positions: np.NDArray[np.float64] or jnp.array, shape (N, D)
            Array of N vertices defining the curve in D-dimensional space.

    Returns:
        distances: jnp.array, shape (M,)
            Array of minimum distances from each data point to the curve.
    """
    # # Ensure inputs are JAX arrays
    # data_points = jnp.asarray(data_points)
    # vertex_positions = jnp.asarray(vertex_positions)

    P1 = vertex_positions[:-1]
    P2 = vertex_positions[1:]

    # Vectorize the distance calculation for all segments
    @jax.vmap
    def point_to_curve(Q):
        segment_distances = jax.vmap(point_to_segment_distance, in_axes=(None, 0, 0))(Q, P1, P2)
        return jnp.min(segment_distances)  # Find the minimum distance to any segment

    return point_to_curve(data_points)


class Material(ABC):
    @abstractmethod
    def energy(
        self, vertex_positions: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute the energy

        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates

        Returns:
            npt.NDArray[np.float64]: Energy
        """
        pass

    @abstractmethod
    def force(
        self, vertex_positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the force

        Args:
            vertex_positions (npt.NDArray[np.float64]): coordinates

        Returns:
            npt.NDArray[np.float64]: Forces
        """
        pass

    @abstractmethod
    def energy_force(
        self, vertex_positions: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute the energy and force

        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates

        Returns:
            npt.NDArray[np.float64]: Energy and forces
        """
        pass

    def _apply_boundary_conditions(
        self, force: npt.NDArray[np.float64], boundary_condition: str
    ) -> npt.NDArray[np.float64]:
        """Apply boundary conditions to the computed force array.

        Args:
            force (npt.NDArray[np.float64]): The computed force array.
            boundary_condition (str): The type of boundary condition (e.g., "pinned", "fixed").

        Returns:
            npt.NDArray[np.float64]: Force array after applying the boundary condition.
        """
        if boundary_condition == "pinned":
            # Fix the first and last vertices
            force = force.at[:, 0, :].set(0.0)   # First vertex
            force = force.at[:, -1, :].set(0.0)  # Last vertex
        elif boundary_condition == "fixed":
            # Fix the first three and last three vertices
            force = force.at[:, :3, :].set(0.0)  # First three vertices
            force = force.at[:, -3:, :].set(0.0) # Last three vertices
        elif boundary_condition == "open":
            pass
        else:
            raise ValueError(f"Unknown boundary condition: {boundary_condition}")

        return force

    def get_energy_shape(
            self, 
            vertex_positions: npt.NDArray[np.float64], 
        ) -> tuple:
        """Get the shape of the energy array.

        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates of the vertices.

        Returns:
            tuple: Shape of the energy array.
        """

        if self.spont_curvatures is None:
            self.spont_curvatures = np.zeros(vertex_positions.shape[0]-1)

        return self._energy(vertex_positions).shape


class ClosedPlaneCurveMaterial(Material):
    def __init__(
        self,
        kappa: float = 0.1,
        sigma: float = 50,
        Kr: float = 1,
        boundary: str = None
    ):
        """Initialize plane curve material

        Args:
            kappa (float, optional): Bending modulus in units of pN um.Defaults to 1.
            sigma (float, optional): Global stretching modulus in units of PN um/um^2. Defaults to 0.
            Kr (float, optional): Regularization modulus. Defaults to 1.
        """
        self.kappa = kappa
        self.sigma = sigma
        self.Kr = Kr
        self.boundary = boundary

    def _check_valid(
        self,
        vertex_positions: npt.NDArray[np.float64],
    ) -> bool:
        """_summary_

        Args:
            vertex_positions (npt.NDArray[np.float64]): _description_

        Raises:
            RuntimeError: Error if first and past points of closed polygon are not the same

        Returns:
            bool: True
        """
        if not np.allclose(vertex_positions[-1], vertex_positions[0]):
            raise RuntimeError(
                f"First ({vertex_positions[0]}) and last ({vertex_positions[-1]}) points are expected to be the same."
            )
        return True

    @partial(jax.jit, static_argnums=0)
    def _energy(
        self,
        vertex_positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the energy of a 2D discrete closed polygon.

        Note that this function assumes that the coordinates of the last point are the same as the first point.

        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates

        Returns:
            npt.NDArray[np.float64]: Componentwise energies of the system
        """
        d_pos = jnp.roll(vertex_positions[:-1], -1, axis=0) - vertex_positions[:-1]
        edgeLengths = jnp.linalg.norm(d_pos, axis=1)
        referenceEdgeLength = jnp.mean(edgeLengths)

        edgeAbsoluteAngles = jnp.arctan2(d_pos[:, 1], d_pos[:, 0])

        vertexTurningAngles = (
            jnp.roll(edgeAbsoluteAngles, -1) - edgeAbsoluteAngles
        ) % (2 * jnp.pi)
        vertexTurningAngles = (vertexTurningAngles + jnp.pi) % (2 * jnp.pi) - jnp.pi

        tan_vertex_turning_angles = jnp.tan(vertexTurningAngles / 2)

        edgeCurvatures = (
            tan_vertex_turning_angles + jnp.roll(tan_vertex_turning_angles, 1)
        ) / edgeLengths

        bendingEnergy = self.kappa * 0.25 * jnp.sum(edgeCurvatures * edgeCurvatures * edgeLengths)
        surfaceEnergy = self.sigma * jnp.sum(edgeLengths)
        regularizationEnergy = self.Kr * jnp.sum(
            ((edgeLengths - referenceEdgeLength) / referenceEdgeLength) ** 2
        )
        return jnp.array([bendingEnergy, surfaceEnergy, regularizationEnergy])

    def energy(
        self, vertex_positions: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute the energy of a 2D discrete closed polygon.

        Note that this function assumes that the coordinates of the last point are the same as the first point.

        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates

        Returns:
            float: Energy of the system
        """
        self._check_valid(vertex_positions)
        return self._energy(vertex_positions)

    @partial(jax.jit, static_argnums=0)
    def _force(
        self,
        vertex_positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        # return jax.grad(f_energy)(vertex_positions)
        return -jax.jacrev(self._energy)(vertex_positions)

    def force(
        self,
        vertex_positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the force of a 2D discrete closed polygon.

        Note that this function assumes that the coordinates of the last point are the same as the first point.

        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates

        Returns:
            float: Energy of the system
        """
        self._check_valid(vertex_positions)
        return self._force(vertex_positions)

    @partial(jax.jit, static_argnums=0)
    def _energy_force(
        self,
        vertex_positions: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        energy, vjp = jax.vjp(self._energy, vertex_positions)
        (force,) = jax.vmap(vjp, in_axes=0)(-1 * jnp.eye(len(energy)))
        return energy, force

    def energy_force(
        self, vertex_positions: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute the energy and force

        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Energy and force
        """
        self._check_valid(vertex_positions)
        return self._energy_force(vertex_positions)



class OpenPlaneCurveMaterial(Material):
    def __init__(
        self,
        kappa: float = 0.1,
        sigma: float = 50,
        Kr: float = 1,
        k_d: float = 1,
        n_degree: int = 2,
        boundary: str = None, 
        spont_curvatures: npt.NDArray[np.float64] = None, 
        data_points: npt.NDArray[np.float64] = None,
    ):
        """Initialize plane curve material

        Args:
            kappa (float, optional): Bending modulus in units of pN nm. Defaults to 1.
            sigma (float, optional): Global stretching modulus in units of PN nm/nm^2. Defaults to 0.
            Kr (float, optional): Regularization modulus. Defaults to 1.
            Kd (float, optional): Data fidelity modulus. Defaults to 1.
        """
        self.kappa = kappa
        self.sigma = sigma
        self.Kr = Kr
        # self.Kd = Kd
        self.k_d = k_d
        self.n_degree = n_degree
        self.boundary = boundary
        self.spont_curvatures = spont_curvatures
        # self.data_points = jnp.asarray(data_points) # Convert data points to JAX array
        self.data_points = data_points


    def _check_valid(
        self,
        vertex_positions: npt.NDArray[np.float64],
    ) -> bool:
        """_summary_

        Args:
            vertex_positions (npt.NDArray[np.float64]): _description_

        Raises:
            RuntimeError: Error if first and past points of closed polygon are not the same

        Returns:
            bool: True
        """
        if vertex_positions.shape[0] != self.spont_curvatures.shape[0] + 1:
            raise ValueError("Length of spont_curvatures+1 must match the number of vertices.")

        return True


    @partial(jax.jit, static_argnums=0)
    def _energy(
        self, 
        vertex_positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the energy of a 2D discrete closed polygon.

        Note that this function assumes that the coordinates of the last point are the same as the first point.

        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates

        Returns:
            float: Energy of the system
        """
        vertex_positions = jnp.asarray(vertex_positions)

        x = vertex_positions[:, 0]
        y = vertex_positions[:, 1]
        dx = jnp.diff(x)
        dy = jnp.diff(y)
        edgeLengths = jnp.sqrt(dx**2 + dy**2)
        edgeAbsoluteAngles = jnp.arctan2(dy, dx)
        referenceEdgeLength = jnp.mean(edgeLengths)

        vertexTurningAngles = (jnp.diff(edgeAbsoluteAngles)) % (2 * jnp.pi)
        vertexTurningAngles = (vertexTurningAngles + jnp.pi) % (2 * jnp.pi) - jnp.pi

        vertexTurningAngles = jnp.append(vertexTurningAngles, vertexTurningAngles[-1])
        vertexTurningAngles = jnp.append(vertexTurningAngles[0], vertexTurningAngles)

        # vertexTurningAngles = jnp.append(vertexTurningAngles, 0.0)
        # vertexTurningAngles = jnp.append(0.0, vertexTurningAngles)

        edgeCurvatures = (
            jnp.tan(vertexTurningAngles[:-1] / 2) + jnp.tan(vertexTurningAngles[1:] / 2)
        ) / edgeLengths

        edgeCurvatures = edgeCurvatures - self.spont_curvatures

        dataDistances = distances_to_curve(self.data_points, vertex_positions)
        # dataDistances = 0.0


        bendingEnergy = self.kappa * 0.25 * jnp.sum(edgeCurvatures * edgeCurvatures * edgeLengths)
        surfaceEnergy = self.sigma * jnp.sum(edgeLengths)

        # regularizationEnergy = self.Kr * jnp.sum(
        regularizationEnergy = self.Kr * jnp.mean(
            ((edgeLengths - referenceEdgeLength) / referenceEdgeLength) ** 2
        )

        # dataDistances = jnp.where(dataDistances < self.dist_c, 0, dataDistances - self.dist_c)
        # dataEnergy = self.Kd * jnp.sum(dataDistances * dataDistances)
        dataEnergy = self.k_d * jnp.mean(jnp.power(dataDistances, self.n_degree))
        # dataEnergy = jnp.mean(dataDistances * dataDistances * dataDistances * dataDistances)

        return jnp.array([bendingEnergy, surfaceEnergy, regularizationEnergy, dataEnergy])




    def energy(
        self, 
        vertex_positions: npt.NDArray[np.float64], 
    ) -> npt.NDArray[np.float64]:
        """Compute the energy of a 2D discret

        Note that this function assumes that the coordinates of the last point are the same as the first point.

        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates

        Returns:
            float: Energy of the system
        """
        if self.spont_curvatures is None:
            self.spont_curvatures = np.zeros(vertex_positions.shape[0]-1)
        self._check_valid(vertex_positions)
        return self._energy(vertex_positions)

    @partial(jax.jit, static_argnums=0)
    def _force(
        self,
        vertex_positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the force of a 2D discrete closed polygon.

        Note that this function assumes that the coordinates of the last point are the same as the first point.

        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates

        Returns:
            float: Energy of the system
        """

        # computed_force = -jax.jacrev(self.energy)(vertex_positions, spont_curvatures)
        computed_force = -jax.jacrev(self.energy)(vertex_positions)
        computed_force = self._apply_boundary_conditions(computed_force, self.boundary)

        return computed_force

    def force(
        self,
        vertex_positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the force of a 2D discrete closed polygon.

        Note that this function assumes that the coordinates of the last point are the same as the first point.

        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates
            spont_curvatures (npt.NDArray[np.float64], optional): Spontaneous curvatures. Defaults to zero.


        Returns:
            float: Energy of the system
        """
        # self._check_valid(vertex_positions)
        return self._force(vertex_positions)

    @partial(jax.jit, static_argnums=0)
    def _energy_force(
        self, vertex_positions: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute the energy and force

        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Energy and force
        """
        energy, vjp = jax.vjp(self.energy, vertex_positions)
        (force,) = jax.vmap(vjp, in_axes=0)(-1 * jnp.eye(len(energy)))
        # jax.debug.print("Force: {}", force)
        force = self._apply_boundary_conditions(force, self.boundary)
        return energy, force

    def energy_force(
        self, vertex_positions: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute the energy and force

        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Energy and force
        """
        self._check_valid(vertex_positions)
        return self._energy_force(vertex_positions)







