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
        Kb: float = 0.1,
        Ksg: float = 50,
        Ksl: float = 1,
        boundary: str = None
    ):
        """Initialize plane curve material

        Args:
            Kb (float, optional): Bending modulus in units of pN um.Defaults to 1.
            Ksg (float, optional): Global stretching modulus in units of PN um/um^2. Defaults to 0.
            Ksl (float, optional): Regularization modulus. Defaults to 1.
        """
        self.Kb = Kb
        self.Ksg = Ksg
        self.Ksl = Ksl
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

        bendingEnergy = self.Kb * jnp.sum(edgeCurvatures * edgeCurvatures * edgeLengths)
        surfaceEnergy = self.Ksg * jnp.sum(edgeLengths)
        regularizationEnergy = self.Ksl * jnp.sum(
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
        Kb: float = 0.1,
        Ksg: float = 50,
        Ksl: float = 1,
        boundary: str = None, 
        spont_curvatures: npt.NDArray[np.float64] = None
    ):
        """Initialize plane curve material

        Args:
            Kb (float, optional): Bending modulus in units of pN um.Defaults to 1.
            Ksg (float, optional): Global stretching modulus in units of PN um/um^2. Defaults to 0.
            Ksl (float, optional): Regularization modulus. Defaults to 1.
        """
        self.Kb = Kb
        self.Ksg = Ksg
        self.Ksl = Ksl
        self.boundary = boundary
        self.spont_curvatures = spont_curvatures


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

        # # TEST spontaneous curvature implementation
        # # edgeCurvatures = edgeCurvatures - 1.0/10 # testing for non-zero \bar{H}
        # curve_str = 40
        # curve_end = 60
        # spont_cvt = 0.3
        # edgeCurvatures = edgeCurvatures.at[curve_str:curve_end].set(edgeCurvatures[curve_str:curve_end] - spont_cvt)
        # ####

        edgeCurvatures = edgeCurvatures - self.spont_curvatures

        bendingEnergy = self.Kb * jnp.sum(edgeCurvatures * edgeCurvatures * edgeLengths)
        surfaceEnergy = self.Ksg * jnp.sum(edgeLengths)

        regularizationEnergy = self.Ksl * jnp.sum(
            ((edgeLengths - referenceEdgeLength) / referenceEdgeLength) ** 2
        )

        return jnp.array([bendingEnergy, surfaceEnergy, regularizationEnergy])
        # return jnp.array([surfaceEnergy, regularizationEnergy])

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






